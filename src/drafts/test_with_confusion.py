def test_with_confusion(network, loader, optimizer, device, set_):
    network.eval()
    test_loss = 0
    correct = 0
    tp=0
    fp=0
    fn=0
    tn=0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        target_int = torch.flatten(target, start_dim=0)

        output = network(data)
        test_loss += F.cross_entropy(output, target_int)

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target_int.data.view_as(pred)).cpu().sum()

        tp_,fp_,tn_,fn_=confusion(pred, target_int.data.view_as(pred))
        tp+=tp_
        fp+=fp_
        fn+=fn_
        tn+=tn_

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=100. * correct / (len(loader.dataset))
    test_loss /= len(loader.dataset)
    print('\n '+set_+ ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        accuracy))
    print('\n Precision : {} and Recall : {} \n'.format(precision,recall))
    scores_dic={'accuracy':accuracy,'tp':tp,'fp':fp,'tn':tn,'fn':fn}
    return(scores_dic)

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives