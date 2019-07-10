addpath C:\Users\Shadman\Documents\Deeplearning\Assignments\Ass1\cifar-10-batches-mat

%% Ex 1
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

mean_X = mean(trainX,2);
trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
validX = validX - repmat(mean_X, [1, size(validX,2)]);
testX = testX - repmat(mean_X, [1, size(testX,2)]);

% number of nodes
m = 50;

[W,b] = init(trainX, trainY, m);

%% Compare gradients
% [grad_W,grad_b] = ComputeGradients(X ,Y, W, b, lambda);
% 
% h = 1e-5;
% 
% [grad_b_NumSlow, grad_W_NumSlow] = ComputeGradsNumSlow(X, Y, W, b, lambda, h);
% 
% 
% grad_W1 = grad_W{1};
% grad_W2 = grad_W{2};
% grad_b1 = grad_b{1};
% grad_b2 = grad_b{2};
% 
% grad_W_NumSlow1 = grad_W_NumSlow{1};
% grad_W_NumSlow2 = grad_W_NumSlow{2};
% grad_b_NumSlow1 = grad_b_NumSlow{1};
% grad_b_NumSlow2 = grad_b_NumSlow{2};
% 
% for i = 1:size(grad_W1,2)
%     sums1(i) = sqrt(sum((grad_W1(:,i)-grad_W_NumSlow1(:,i)).^2));
% end
% compare_grad1W = sum(sums1)
% 
% compare_grad1b = sqrt(sum((grad_b1-grad_b_NumSlow1).^2))
% 
% for i = 1:size(grad_W2,2)
%     sums2(i) = sqrt(sum((grad_W2(:,i)-grad_W_NumSlow2(:,i)).^2));
% end
% compare_grad2W = sum(sums2)
% 
% compare_grad2b = sqrt(sum((grad_b2-grad_b_NumSlow2).^2))


%% Momentum time comparison on training

% n_batch = 100;
% eta = 0.1;
% n_epochs = 50;
% rho = 0.9;
% lambda = 0;
% 
% tic
% [~, ~, loss_tra_mom, ~,~,~] = MiniBatchGDMomentum(trainX, trainY,0, validX, validY,0, n_batch, eta, n_epochs, W, b, lambda, rho);
% toc
% 
% tic
% [~, ~, loss_tra, ~,~,~] = MiniBatchGDMomentum(trainX, trainY,0, validX, validY,0, n_batch, eta, n_epochs, W, b, lambda, 0);
% toc   
% 
% plot(loss_tra_mom);
% hold on;
% plot(loss_tra);
% grid on;
% legend('Training loss with momentum','Training loss with no momentum');
% xlabel('Eps');
% ylabel('Loss Function');

%% Gridsearch

% Coarse search
% n_batch = 100;
% n_epochs = 5;
% rho = 0.9; %0.5 0.9 0.99
% 
% l_min = -8; 
% l_max = 0;
% e_min= -3;
% e_max= -1;
% 
% for i=1:50
%     [W,b] = initialize(trainX, trainY, m);
%     l = l_min + (l_max - l_min)*rand(1, 1);
%     lambda(i) = 10^l;
%     e = e_min + (e_max - e_min)*rand(1, 1);
%     eta(i) = 10^e;
% 
%     [Wstar_t, bstar_t, ~,~] = MiniBatchGDMomentum(trainX, trainY,validX, validY, n_batch, eta(i), n_epochs, W, b, lambda(i), rho);
% 
%     acc_valid(i) = ComputeAccuracy(validX, validy, Wstar_t, bstar_t);
% end
% 
% [max_value,max_index] = max(acc_valid);
% lam_max = lambda(max_index); 
% eta_max = eta(max_index);
% 
% acc_valid(max_index) = NaN;
% 
% [max_value_next,next_max_index] = max(acc_valid);
% lam__next_max = lambda(next_max_index); 
% eta_next_max = eta(next_max_index);
% 
% acc_valid(next_max_index) = NaN;
% 
% [max_value_third,third_max_index] = max(acc_valid);
% lam__third_max = lambda(third_max_index); 
% eta_third_max = eta(third_max_index);
% 
% acc_valid(max_index) = max_value;
% acc_valid(next_max_index) = max_value_next;
% 
% e_min1 = log10(eta_next_max);
% e_max1= log10(eta_max);
% 
% l_min1 = log10(lam__next_max); 
% l_max1 = log10(lam_max); 
% 
% Fine search
% for j=1:50
%     [W,b] = initialize(trainX, trainY, m);
%     l = l_min1 + (l_max1 - l_min1)*rand(1, 1);
%     lambda1(j) = 10^l;
%     e = e_min1 + (e_max1 - e_min1)*rand(1, 1);
%     eta1(j) = 10^e;
% 
%     [Wstar_t, bstar_t, ~,~] = MiniBatchGDMomentum(trainX, trainY,validX, validY, n_batch, eta1(j), n_epochs, W, b, lambda1(j), rho);
% 
%     acc_valid1(j) = ComputeAccuracy(validX, validy, Wstar_t, bstar_t);
% end
% 
% [max_value1,max_index1] = max(acc_valid1);
% lam_max1 = lambda1(max_index1);  #Your best lambda
% eta_max1 = eta1(max_index1);     #Your best eta
% 
% acc_valid1(max_index1) = NaN;
% 
% [max_value_next1,next_max_index1] = max(acc_valid1);
% lam__next_max1 = lambda1(next_max_index1); 
% eta_next_max1 = eta1(next_max_index1);
% 
% acc_valid1(next_max_index1) = NaN;
% 
% [max_value_third1,third_max_index1] = max(acc_valid1);
% lam__third_max1 = lambda1(third_max_index1); 
% eta_third_max1 = eta1(third_max_index1);
% 
% acc_valid1(max_index1) = max_value1;
% acc_valid1(next_max_index1) = max_value_next1;

%% Run with best param and minibatch
n_batch = 100;
eta = 0.0342; %eta_max1
n_epochs = 1;
rho = 0.9;
lambda = 2.7653e-07; %lam_max1

[W_opt, b_opt, loss_tra, loss_val, accuracy_train, accuracy_valid] = MBGDM(trainX, trainY,trainy,  validX, validY,validy, n_batch, eta, n_epochs, W, b, lambda, rho);


accuracy_train_last = CoA(trainX, trainy, W_opt, b_opt)
accuracy_valid_last = CoA(validX, validy, W_opt, b_opt)
accuracy_test = CoA(testX, testy, W_opt, b_opt)


plot(loss_tra);
hold on;
plot(loss_val);
grid on;
legend('training loss','validation loss');
xlabel('epoch');
ylabel('loss');

%% functions

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

function [W, b] = init(trainX, trainY, m)
He1 = sqrt(2/size(trainX,1));
He2 = sqrt(2/m);

W1 = 0+He1*randn(m,size(trainX,1));
W2 = 0+He2*randn(size(trainY,1),m);

b1 = zeros(m,1);
b2 = zeros(size(trainY,1),1);

W = {W1,W2};
b = {b1,b2};
end

function [P] = EvC(X, W, b)


W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};

P = zeros(size(b2,1),size(X,2));


for i = 1:size(X,2)
x = X(:,i);
s1 = W1*x + b1;
h = max(0,s1);
%h = tanh(s1);
%h = 1./(1+exp(-s1));
s2 = W2*h+b2;

soft = exp(s2)/(ones(1,size(s2,1))*exp(s2));

P(:,i) = soft;

end

end

function cost = CoC(X,Y,W,b,lambda)
W1 = W{1};
W2 = W{2};

prob = EvC(X,W,b);
PY = Y.*prob;
PY = sum(PY);
LCross = -log(PY);
reg1 = lambda*sum(sum(W1.^2));
reg2 = lambda*sum(sum(W2.^2));
cost = sum(LCross )/size(X,2) + reg1 + reg2;

end

function acc = CoA(X, y, W, b)
prob = EvC(X, W, b);

c = 0;
for i = 1:size(y,2)
[~,maxindex(i)] = max(prob(:,i));
    if y(i) == maxindex(i)
          c = c+1;
    end
end
acc = c/size(y,2);
end


function [grad_W,grad_b ] = CoG(X ,Y, W,b, lambda)
P = EvC(X,W,b);

dLdb1 = 0;
dLdW1 = 0;
dLdb2 = 0;
dLdW2 = 0;

W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};

for i = 1: size(X,2)
x = X(:,i);    
gi = -(Y(:,i)-P(:,i))';
dLdb2 = dLdb2 + gi';
si = W1*x + b1;
hi = max(0,si);
dLdW2 = dLdW2 + gi'*hi';
gi = gi*W2;
gi = gi*diag(si>0);
dLdb1 = dLdb1 + gi';

dLdW1 = dLdW1 +gi'*x';

end

dLdb2 = dLdb2/size(X,2);
dLdW2 = dLdW2/size(X,2);
dLdb1 = dLdb1/size(X,2);
dLdW1 = dLdW1/size(X,2);

grad_W1=dLdW1+2*lambda*W1;
grad_W2=dLdW2+2*lambda*W2;
grad_W={grad_W1,grad_W2};

grad_b1=dLdb1;
grad_b2=dLdb2;
grad_b={grad_b1,grad_b2};
end

function [W, b, CostTrain, CostValid, accuracy_train, accuracy_valid] = MBGDM(trainX, trainY, trainy, validX, validY,validy, n_batch, eta, n_epochs, W, b, lambda, rho)

vW1 = 0;
vW2 = 0;
vb1 = 0;
vb2 = 0;

for i = 1: n_epochs
    for j = 1:size(trainX,2)/n_batch
        j_start = (j-1)*n_batch+1;
        j_end = j*n_batch;
        Xbatch = trainX(:, j_start:j_end);
        Ybatch = trainY(:,j_start:j_end);
    
        [grad_W,grad_b] = CoG(Xbatch ,Ybatch, W,b, lambda);
        
        vW1 = rho*vW1 + eta*grad_W{1};
        W{1} = W{1} - vW1;
        
        vW2 = rho*vW2 + eta*grad_W{2};
        W{2} = W{2} - vW2;
        
        vb1 = rho*vb1 + eta*grad_b{1};
        b{1} = b{1} - vb1;
        
        vb2 = rho*vb2 + eta*grad_b{2};
        b{2} = b{2} - vb2;
        
    end
     if i > 4
       eta = eta/10;
     end
    CostTrain(i) = CoC(trainX,trainY,W,b,lambda);
    CostValid(i) = CoC(validX,validY,W,b,lambda);
    accuracy_train(i) = CoA(trainX, trainy, W, b);
    accuracy_valid(i) = CoA(validX, validy, W, b);
    
end

end