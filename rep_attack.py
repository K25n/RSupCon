import torch
from losses import SupConLoss_robust



def project(x, original_x, epsilon):


    max_x = original_x + epsilon
    min_x = original_x - epsilon

    x = torch.max(torch.min(x, max_x), min_x)

    return x



class SupConRepAdv:

    def __init__(
        self,
        model,
        epsilon=8 / 255,
        alpha=2 / 255,
        steps=7,
        atk_temp=0.1
    ):

        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.atk_temp = atk_temp

    def perturb(
        self, source_images_list, target_images_list, labels, rand_idx = None
    ):

        bsz = labels.shape[0]

        adv_images_list = []

        for i, source_images in enumerate(source_images_list):
            if rand_idx is not None and i in rand_idx:
                
                perturb_shape = source_images.shape
                rand_perturb = (
                    torch.FloatTensor(perturb_shape)
                    .uniform_(-self.epsilon, self.epsilon)
                    .float()
                    .cuda()
                )
                adv_images = source_images.float().clone() + rand_perturb
            else:
                adv_images = source_images.float().clone()

            adv_images_list.append(torch.clamp(adv_images, 0, 1))

        source_images = torch.cat(source_images_list, dim=0)
        target_images = torch.cat(target_images_list, dim=0)
        adv_images = torch.cat(adv_images_list, dim=0)

        adv_images.requires_grad = True

        loss = SupConLoss_robust(
            temperature=self.atk_temp,
            
        )
        self.model.eval()
        
        target_f_split = torch.split(self.model(target_images), bsz, dim=0)
        target_f = torch.cat([f.unsqueeze(1) for f in target_f_split], dim=1)
        
        with torch.enable_grad():
            for _ in range(self.steps):

                self.model.zero_grad()

                adv_f = self.model(adv_images)
                adv_f_split = torch.split(adv_f, bsz, dim=0)
                adv_f_cat = torch.cat([f.unsqueeze(1) for f in adv_f_split], dim=1)

                anchor_adv = adv_f_cat
                contrast_adv = target_f

                
                cost = loss(anchor_adv, contrast_adv, labels=labels)


                grads = torch.autograd.grad(
                    cost,
                    adv_images,
                    grad_outputs=None,
                    only_inputs=True,
                    retain_graph=False,
                )[0]

                scaled_g = torch.sign(grads.data)

                adv_images.data += self.alpha * scaled_g
                adv_images = torch.clamp(adv_images, 0, 1)
                adv_images = project(adv_images, source_images, self.epsilon)

        self.model.train()

        return adv_images.detach()

   