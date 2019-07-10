% Ex 0
addpath C:\Users\Shadman\Documents\Deeplearning\Assignments\Ass1\cifar-10-batches-mat
% 
% A = load('data_batch_1.mat');
% I = reshape(A.data', 32,32,3,10000);
% I = permute(I , [2,1,3,4]);
% 
% montage(I(:,:,:,1:500),'Size',[5,5]);

%% Ex 1, Data
test_batch = load('test_batch.mat');

[X1, Y1, y1] = LoadBatch(load('data_batch_1.mat'));
[X2, Y2, y2] = LoadBatch(load('data_batch_2.mat'));
[X3, Y3, y3] = LoadBatch(load('data_batch_3.mat'));
[X4, Y4, y4] = LoadBatch(load('data_batch_4.mat'));
[X5, Y5, y5] = LoadBatch(load('data_batch_5.mat'));

X = [X1, X2, X3, X4, X5];
Y = [Y1, Y2, Y3, Y4, Y5];
y = [y1, y2, y3, y4, y5];

trainX = X(:,[1:length(X)-1000]);
validX = X(:,[length(X)-999:length(X)]);

trainy = y(:,[1:length(y)-1000]);
validy = y(:,[length(y)-999:length(y)]);

trainY = Y(:,[1:length(Y)-1000]);
validY = Y(:,[length(Y)-999:length(Y)]);

[testX, testY, testy] = LoadBatch(test_batch);

%% Ex 2, init weight, bias
W = 0+randn(size(trainY,1),size(trainX,1));
 
b = 0+randn(size(trainY,1),1);

%Xavier init
W = W*(1/sqrt(size(trainX,1)));

%% EX 7
n_batch = 100;
eta = 0.01;
n_epochs = 5;
lambda = 0;

GDparams = [n_batch, eta, n_epochs];

[W_opt, b_opt, loss_tra, loss_val] = MiniBatchGD(trainX, trainY, trainy, validX, validY,validy ,GDparams, W, b, lambda);


accuracy_train_last = ComputeAccuracy(trainX, trainy, W_opt, b_opt)
accuracy_valid_last = ComputeAccuracy(validX, validy, W_opt, b_opt)
accuracy_test = ComputeAccuracy(testX, testy, W_opt, b_opt)

% Draw loss picture
figure(1)
plot(loss_tra);
hold on;
plot(loss_val);
grid on;
legend('training loss','validation loss');
xlabel('epoch');
ylabel('loss');

% weight image
figure(2);
for i = 1:10
    im = reshape(W(i,:),32,32,3);
    s_im{i} = (im-min(im(:)))/(max(im(:))-min(im(:)));
    s_im{i} = permute(s_im{i},[2,1,3]);
    subplot(1,10,i)
    imshow(s_im{i})
end

% Avmarkera Ex 3 - Ex 6 stegvis, för att checka om du har gjort rätt enligt Ex 3 - Ex 6 i labbpeken 
%% Ex 3
% 
% P = EvaluateClassifier(trainX(:,1:100),W,b);
% 
%% EX 4
% 
% lambda = 1; 
% cost = ComputeCost(trainX(:, 1:100), trainY(:, 1:100), W, b, lambda);
% 
%% EX 5
% 
% acc = ComputeAccuracy(trainX(:, 1:100), trainy(1:100), W, b);
% 
%% EX 6 get gradients
% X = trainX(:, 1:100);
% Y = trainY(:, 1:100);
% lambda = 0;
% [grad_W,grad_b] = ComputeGradients(X ,Y, P, W, lambda);
% 
%% EX 6 get num gradients
% h = 1e-6;
% X = trainX(:, 1:100);
% Y = trainY(:, 1:100);
% lambda = 0;
% 
% [grad_b_NumSlow, grad_W_NumSlow] = ComputeGradsNumSlow(X, Y, W, b, lambda, h);
% 
%% EX 6 compare with num gradient
% h = 1e-6;
% X = trainX(:, 1:1000);
% Y = trainY(:, 1:1000);
% lambda = 0;
% 
% [grad_b_Num, grad_W_Num] = ComputeGradsNum(X, Y, W, b, lambda, h);
% 
% % Compare gradients
% for i = 1:size(grad_W,2)
%     sums(i) = sqrt(sum((grad_W(:,i)-grad_W_NumSlow(:,i)).^2));
% end
% compare_grad1W = sum(sums)
% 
% compare_grad1b = sqrt(sum((grad_b-grad_b_NumSlow).^2))
% 
% for i = 1:size(grad_W,2)
%     sums1(i) = sqrt(sum((grad_W(:,i)-grad_W_Num(:,i)).^2));
% end
% compare_grad2W = sum(sums1)
% 
% compare_grad2b = sqrt(sum((grad_b-grad_b_Num).^2))


%% Functions

function [X,Y,y] = LoadBatch(filname)
X = filname.data';
X = double(X);
X = X/255;

labelMatrix = filname.labels';
labelMatrix = labelMatrix +1;
labelMatrix = double(labelMatrix);
Y = full(ind2vec(labelMatrix));

y = filname.labels';
y = y + 1;
y = double(y);

end

function prob = EvaluateClassifier(X, W, b)

prob = zeros(size(b,1),size(X,2));

for i = 1:size(X,2)
x = X(:,i);
s = W*x + b;
soft = exp(s)/(ones(1,size(s,1))*exp(s));

prob(:,i) = soft;

end

end

function cost = ComputeCost(X,Y,W,b,lambda)

prob = EvaluateClassifier(X,W,b);
PY = Y.*prob;
PY = sum(PY);
LCross = -log(PY);

reg = lambda*sum(sum(W.^2));

cost = sum(LCross )/size(X,2) + reg;

end

function acc = ComputeAccuracy(X, y, W, b)
prob = EvaluateClassifier(X, W, b);

c = 0;
for i = 1:size(y,2)
[~,maxindex(i)] = max(prob(:,i));
    if y(i) == maxindex(i)
          c = c+1;
    end
end
acc = c/size(y,2);
end

function [grad_W,grad_b] = ComputeGradients(X ,Y, P, W, lambda)

dLdw = 0;
dLdb = 0;

for i = 1:size(Y,2)
    
dLdb = dLdb + Y(:,i)-P(:,i);
end

dLdb = -dLdb';

for j = 1:size(X,2)
    x = X(:,j);
dLdw = dLdw - (Y(:,j)-P(:,j))*x';
end

dLdb = dLdb/size(X,2);
dLdw = dLdw/size(X,2);


grad_W=dLdw+2*lambda*W;
grad_b=dLdb;


end

function [W, b, CostTrain, CostVal,accuracy_train, accuracy_valid] = MiniBatchGD(trainX, trainY,trainy, validX, validY, validy, GDparams, W, b, lambda)
n_batch = GDparams(1);
eta = GDparams(2);
n_epochs = GDparams(3); 

for i = 1: n_epochs
    for j = 1:length(trainX)/n_batch
        j_start = (j-1)*n_batch+1;
        j_end = j*n_batch;
        Xbatch = trainX(:, j_start:j_end);
        Ybatch = trainY(:,j_start:j_end);
        
        prob = EvaluateClassifier(Xbatch, W, b);
    
        [grad_W,grad_b] = ComputeGradients(Xbatch ,Ybatch, prob, W, lambda);

        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end
    eta = eta*0.9;
    CostTrain(i) = ComputeCost(trainX,trainY,W,b,lambda);
    CostVal(i) = ComputeCost(validX,validY,W,b,lambda);
    accuracy_train(i) = ComputeAccuracy(trainX, trainy, W, b);
    accuracy_valid(i) = ComputeAccuracy(validX, validy, W, b);
end

end