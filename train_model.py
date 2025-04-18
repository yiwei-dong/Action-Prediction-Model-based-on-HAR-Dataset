import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import logging
import os

def training_model(args, model, train_dataloader, test_dataloader):
    if args.use_gpu:
        device = torch.device(f'cuda:{args.gpu_num}')
    else:
        device = torch.device('cpu')

    learning_rate = args.learning_rate

    writer = SummaryWriter(args.log_dir)

    # log
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('ActionPrediction.log'), logging.StreamHandler()])

    logging.info('Action Prediction Training Started')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    Best_Accuracy_Rate = 0.0
    logging.info("Training!!!")
    for epoch in range(args.epochs):
        train_loss, Train_Accuracy_Rate = train(model, train_dataloader, learning_rate, criterion, optimizer, device)
        test_loss, Test_Accuracy_Rate = evaluate(model, test_dataloader, criterion, device)

        scheduler.step(Test_Accuracy_Rate)

        logging.info(
            'Train Epoch [{}/{}], Loss: {:.8f}, Accuracy: {:.2f}%'.format(epoch + 1, args.epochs, train_loss, Train_Accuracy_Rate*100))
        writer.add_scalar('Train_Loss', train_loss, epoch + 1)
        writer.add_scalar('Train_Accuracy_Rate', Train_Accuracy_Rate*100, epoch + 1)

        logging.info(
            'Test Epoch [{}/{}], Loss: {:.8f}, Accuracy: {:.2f}%'.format(epoch + 1, args.epochs, test_loss, Test_Accuracy_Rate*100))
        writer.add_scalar('Test_Loss', test_loss, epoch + 1)
        writer.add_scalar('Test_Accuracy_Rate', Test_Accuracy_Rate*100, epoch + 1)

        if Test_Accuracy_Rate > Best_Accuracy_Rate:
            Best_Accuracy_Rate = Test_Accuracy_Rate
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir,'best_model.pth'))
    logging.info(
        'Best Test Accuracy Rate: {:.2f}'.format(Best_Accuracy_Rate*100))
    logging.info("Ending, best model was saved at {}/best_model.pth".format(args.save_dir))
    writer.close()



def train(model, dataloader, learning_rate, criterion, optimizer, device):
    total_loss = 0
    correct = 0
    model.train()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad() # 在每次迭代开始时，需要清零梯度，否则梯度会累积
        # print("Input shape:", inputs.shape)  # 应该是 [batch_size, input_dimension]
        outputs = model(inputs)
        # print("Output shape:", outputs.shape)  # 应该是 [batch_size, num_classes]
        # print("labels shape:", labels.shape)  # 应该是 [batch_size]
        loss = criterion(outputs, labels) # 计算损失
        loss.backward() # 计算损失对模型参数的梯度
        optimizer.step() # 根据优化器的规则（如学习率、动量等）更新模型参数
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += torch.sum(predicted == labels.data).item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels.data).item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)