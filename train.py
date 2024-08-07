import torch
import os
from tqdm import tqdm
from segment_anything.build_sam import build_siamese_sam
from segment_anything.dataloader import MedicalDataset
from segment_anything.utils import ContrasiveStructureLoss, LaplacianPyramid
from collections import OrderedDict


if __name__ == '__main__':
    epochs = 400
    batch_size = 2
    lr = 1e-6
    device = "cuda"
    checkpoint = 'model/best/SiameseSAM_epoch135.pth'

    print('-'*15,'Loading Data','-'*15)
    medical_dataset = MedicalDataset(root='dataset', mod1='CT', mod2='MR-T2')
    num_classes = medical_dataset.num_classes
    mean, std, total_samples = medical_dataset.mean_std()
    print('Mean: ', mean)
    print('Std: ', std)
    print('Total Samples: ', total_samples)
    data_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print('Finished!')
    print('-'*15,'Init Model','-'*15)
    SiameseSAM = build_siamese_sam(num_classes=num_classes, checkpoint=checkpoint).to(device)
    print('Load model from', checkpoint)

    laplacian_pyramid = LaplacianPyramid(levels=4, device=device).to(device)
    # SiameseSAM = torch.nn.DataParallel(SiameseSAM, [0,1,2,3])

    # for name, param in SiameseSAM.named_parameters():
    #     if name in 'mask_decoder':
    #         param.requires_grad = False
    if checkpoint is None:
        pretrained_dict = torch.load('sam_vit_b_01ec64.pth')
        current_state_dict = SiameseSAM.state_dict()
        # 加载权重时使用 strict=False 参数

        skip_weights = [
        'image_encoder.patch_embed.proj.weight',
        'prompt_encoder.mask_downscaling.0.weight',
        'prompt_encoder.mask_downscaling.3.weight',
        'mask_decoder.mask_tokens.weight'
        ]
        new_state_dict = OrderedDict()
        for name, param in pretrained_dict.items():
            if name in skip_weights:
                print(f'Skipping {name} as it is in the skip list')
                continue
            if name in current_state_dict:
                if current_state_dict[name].shape == param.shape:
                    new_state_dict[name] = param
                else:
                    print(f'Skipping {name} due to size mismatch: '
                        f'{param.shape} (checkpoint) vs {current_state_dict[name].shape} (current model)')
            else:
                print(f'Skipping {name} as it is not found in the current model')
        current_state_dict.update(new_state_dict)
        SiameseSAM.load_state_dict(current_state_dict, strict=False)
    for name, param in SiameseSAM.named_parameters():
        if 'prompt_encoder.clip' in name:
            param.requires_grad = False
    criterion = ContrasiveStructureLoss(device=device)
    optimizer = torch.optim.Adam(SiameseSAM.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(SiameseSAM.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    print('Finished!')
    print('-'*15,'Training','-'*15)
    best_loss = 1.9
    for epoch in range(epochs):
        mean_loss = 0
        for inputs, labels in tqdm(data_loader):
            # laplacian_CT, laplacian_MRI = laplacian_pyramid.build_laplacian_pyramid_CT(inputs[0]), laplacian_pyramid.build_laplacian_pyramid_MRI(inputs[1])
            # inputs = [laplacian_CT[0], laplacian_MRI[0]]

            inputs, labels = [x.to(device) for x in inputs], [y.to(device) for y in labels]
            optimizer.zero_grad()
            outputs = SiameseSAM(inputs)
            outputs = [i.to(device) for i in outputs]
            # outputs[4] = laplacian_pyramid(outputs[4])
            loss = criterion(CT_pred=outputs[0],
                             MRI_pred=outputs[1],
                             merged=outputs[4],
                             encoded_CT=outputs[2],
                             encoded_MRI=outputs[3],
                             CT_target=labels[0],
                             MRI_target=labels[1],
                             origin_CT=inputs[0],
                             origin_MRI=inputs[1])
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
            # for param in SiameseSAM.parameters():
            #     print(param.grad)
        scheduler.step()
        mean_loss /= len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss}")
        if (loss < best_loss):
            torch.save(SiameseSAM.state_dict(), f'model/SiameseSAM_best_epoch{epoch+1}.pth')
            best_loss = loss
            print('Best Loss: ', best_loss)
        if ((epoch+1) % 5 == 0):
            torch.save(SiameseSAM.state_dict(), f'model/SiameseSAM_epoch{epoch+1}.pth')
