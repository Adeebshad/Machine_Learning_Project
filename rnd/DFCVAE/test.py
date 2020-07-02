import torch


def test(model, device, test_loader, epoch):
    original_images, rect_images = [], []
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, y = sample['image'].to(device),\
                sample['attributes'].to(device)
            data = data.view((-1, 3, 64, 64))
            y = y.view(-1, 3)
            output, mu, logvar = model(data, y)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()
            original_images.append(data[0].view((64, 64, 3)).cpu())
            rect_images.append(output[0].view((64, 64, 3)).cpu())
            if batch_idx == 4:
                break

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)
