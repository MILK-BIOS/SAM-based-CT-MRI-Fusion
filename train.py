import torch
from tqdm import tqdm
from segment_anything.build_sam import build_siamese_sam
from segment_anything.dataloader import MedicalDataset
from segment_anything.utils import ContrasiveStructureLoss


if __name__ == '__main__':
    epochs = 100
    batch_size = 2
    lr = 1e-5
    device = "cuda"

    print('-'*15,'Loading Data','-'*15)
    medical_dataset = MedicalDataset(root='dataset', mod1='CT', mod2='MR-T2')
    num_classes = medical_dataset.num_classes
    data_loader = torch.utils.data.DataLoader(medical_dataset, batch_size=batch_size, shuffle=True)
    print('Finished!')
    print('-'*15,'Init Model','-'*15)
    SiameseSAM = build_siamese_sam(num_classes=num_classes, checkpoint=None).to(device)
    SiameseSAM = torch.nn.DataParallel(SiameseSAM, [0,1,2,3])

    criterion = ContrasiveStructureLoss(device=device)
    optimizer = torch.optim.SGD(SiameseSAM.parameters(), lr=0.001, momentum=0.9)
    print('Finished!')
    print('-'*15,'Training','-'*15)
    best_loss = 1.9
    for epoch in range(epochs):
        for inputs, labels in tqdm(data_loader):
            inputs, labels = [x.to(device) for x in inputs], [y.to(device) for y in labels]
            optimizer.zero_grad()
            outputs = SiameseSAM(inputs)
            outputs = [i.to(device) for i in outputs]
            # outputs[0] = outputs[0].argmax(dim=-1).float()
            # outputs[1] = outputs[1].argmax(dim=-1).float()
            loss = criterion(CT_pred=outputs[0],
                             MRI_pred=outputs[1],
                             merged=outputs[2],
                             encoded_CT=outputs[3],
                             encoded_MRI=outputs[4],
                             CT_target=labels[0],
                             MRI_target=labels[1],
                             origin_CT=inputs[0],
                             origin_MRI=inputs[1])
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        if (loss < best_loss):
            torch.save(SiameseSAM.state_dict(), f'model/SiameseSAM_best_epoch{epoch+1}.pth')
            best_loss = loss
        if (epoch % 5 == 0):
            torch.save(SiameseSAM.state_dict(), f'model/SiameseSAM_epoch{epoch+1}.pth')
