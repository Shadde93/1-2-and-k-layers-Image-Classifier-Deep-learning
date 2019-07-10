addpath C:\Users\Shadman\Documents\Deeplearning\Assignments\Ass1\cifar-10-batches-mat

test_batch = load('test_batch.mat');

[X1, Y1, y1] = LoB(load('data_batch_1.mat'));
[X2, Y2, y2] = LoB(load('data_batch_2.mat'));
[X3, Y3, y3] = LoB(load('data_batch_3.mat'));
[X4, Y4, y4] = LoB(load('data_batch_4.mat'));
[X5, Y5, y5] = LoB(load('data_batch_5.mat'));

X = [X1, X2, X3, X4, X5];
Y = [Y1, Y2, Y3, Y4, Y5];
y = [y1, y2, y3, y4, y5];

trainX = X(:,[1:length(X)-1000]);
validX = X(:,[length(X)-999:length(X)]);

trainy = y(:,[1:length(y)-1000]);
validy = y(:,[length(y)-999:length(y)]);

trainY = Y(:,[1:length(Y)-1000]);
validY = Y(:,[length(Y)-999:length(Y)]);

[testX, testY, testy] = LoB(test_batch);

mean_X = mean(trainX,2);
trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
validX = validX - repmat(mean_X, [1, size(validX,2)]);
testX = testX - repmat(mean_X, [1, size(testX,2)]);
%----------------------------------------------------------------
% Put number of layers with a number of nodes
m = [size(trainX,1), 50,30, size(trainY,1)];

[W,b] = init(m);

%% gradient comparison
%
% X = trainX(:, 1:100);
% Y = trainY(:, 1:100);
% y = trainy(:, 1:100);
% lambda = 0;
% 
% 
% [grad_W,grad_b] = ComputeGradients(X ,Y, W, b, lambda);
% 
% h_lim = 1e-5;
% 
% [grad_b_NumSlow, grad_W_NumSlow] = ComputeGradsNumSlow(X, Y, W, b, lambda, h_lim);
% 
% 
% for j = 1:length(grad_W)
% grad_W1 = grad_W{j};
% grad_W1Slow = grad_W_NumSlow{j};
% 
% for i = 1:size(grad_W1,2)
%     sums(i) = sqrt(sum((grad_W1(:,i)-grad_W1Slow(:,i)).^2));
% end
% 
% compare_gradW(j) = sum(sums);
% 
% end
% 
% for k = 1:length(grad_b)
%     
% compare_gradb(k) = sqrt(sum((grad_b{k}-grad_b_NumSlow{k}).^2));
% 
% end
% 
% compare_gradW
% 
% compare_gradb


%% EX 4 lambda and eta

% n_batch = 100;
% n_epochs = 5;
% rho = 0.9;
% 
% l_min = -5; 
% l_max = 0;
% e_min= -3;
% e_max= 2;
% 
% for i=1:25
%     m = [size(trainX,1), 50,30, size(trainY,1)];
% 
%     [W,b] = initialize(m);
%     l = l_min + (l_max - l_min)*rand(1, 1);
%     lambda(i) = 10^l;
%     e = e_min + (e_max - e_min)*rand(1, 1);
%     eta(i) = 10^e;
% 
%     [Wstar_t, bstar_t, ~,~] = MiniBatchGDMomentum(trainX, trainY,trainy,validX, validY,validy, n_batch, eta(i), n_epochs, W, b, lambda(i), rho);
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
% for j=1:25
%     
%     m = [size(trainX,1), 50,30, size(trainY,1)];
% 
%     [W,b] = initialize(m);
%     
%     l = l_min1 + (l_max1 - l_min1)*rand(1, 1);
%     lambda1(j) = 10^l;
%     e = e_min1 + (e_max1 - e_min1)*rand(1, 1);
%     eta1(j) = 10^e;
% 
%     [Wstar_t, bstar_t, ~,~] = MiniBatchGDMomentum(trainX, trainY,trainy,validX, validY,validy, n_batch, eta1(j), n_epochs, W, b, lambda1(j), rho);
% 
%     acc_valid1(j) = ComputeAccuracy(validX, validy, Wstar_t, bstar_t);
% end
% 
% [max_value1,max_index1] = max(acc_valid1);
% lam_max1 = lambda1(max_index1); #your best lambda
% eta_max1 = eta1(max_index1);  #your best eta
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

%% eta hig, medium, low -plots
% n_batch = 100;
% 
% n_epochs = 10;
% 
% 
% lambda = 1.0e-5;
% rho = 0.9;
% 
% eta = [10^-4, 10^-2, 1];
% 
% 
% for i = 1:length(eta)
% [Wstar, bstar, Jtrain{i}, Jval{i}] = MiniBatchGDMomentum(trainX, trainY, validX,validY, n_batch, eta(i), n_epochs, W, b, lambda, rho);
% 
% end
% 
% ComputeAccuracy(trainX, trainy, Wstar, bstar)
% ComputeAccuracy(validX, validy, Wstar, bstar)
% ComputeAccuracy(testX, testy, Wstar, bstar)
% 
% plot(Jtrain{3})
% hold on
% plot(Jval{3})
% grid on;
% legend('training loss','val loss');
% xlabel('Eps');
% ylabel('Loss Function');

%% main run with hyper param
n_batch = 100;
n_epochs = 2;

lambda = 5.9080e-05;
rho = 0.9;
eta = 0.0473;
alph = 0.99; 

[W_opt, b_opt, Jtrain, Jval, accTrain,accValid, my_av, v_av] = MBGDM(trainX, trainY,trainy, validX,validY,validy, n_batch, eta, n_epochs, W, b, lambda, rho, alph);


CoA(trainX, trainy, W_opt, b_opt, my_av, v_av)
CoA(validX, validy, W_opt, b_opt, my_av, v_av)
CoA(testX, testy, W_opt, b_opt, my_av, v_av)

plot(Jtrain)
hold on
plot(Jval)
grid on;
legend('training loss','val loss');
xlabel('Eps');
ylabel('Loss Function');

%% Functions

function [X,Y,y] = LoB(filname)

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

function [W, b] = init(m)

leng = length(m);

W = {};
b = {};

for i = 1:leng-1
W{i} = 0+0.001*randn(m(i+1),m(i));
b{i} = zeros(m(i+1),1);

end
end

function [s_hat] = BatchNorm(s,my,v)

s_hat = (diag(v+(10^(-6)))^(-1/2))*(s-my);
    
end

function [P, S, my, v, s_hat,h] = EvC(X, W, b, my_av,v_av)

x = X;
my = {};
v = {};
S = {};
s_hat = {};
h = {};

if nargin < 4
    for j = 1:length(W)
    
        if j == length(W)
        s = W{j}*x + b{j};
        else
        
    s = W{j}*x + b{j};
    my{j} = mean(s,2);
    v{j} = var(s,0,2)*((size(X,2))-1)/size(X,2);
    s_hat{j} = BatchNorm(s,my{j},v{j});
    x = max(0,s_hat{j});
    
    h{j} = x;
    S{j} = s;

        end
    end
else
    for i = 1:length(W)
        
        if i == length(W)
        s = W{i}*x + b{i};
        else
            
        s = W{i}*x + b{i};
        s_hat_1 = BatchNorm(s,my_av{i},v_av{i});
        x = max(0,s_hat_1);
        end
    end
end

P = softmax(s);
end

function J = CoC(X,Y,W,b,lambda, my_av, v_av)

if nargin < 6
[P,~,~,~,~] = EvC(X,W,b);
else
[P,~,~,~,~] = EvC(X,W,b,my_av, v_av);
end

LCross = -log(sum(Y.*P,1));

for i = 1: length(W)
reg(i) = lambda*sum(sum(W{i}.^2));
end

J = sum(LCross )/size(X,2) + sum(reg);

end


function acc = CoA(X, y, W, b, my_av, v_av)

if nargin < 6
[P,~,~,~,~] = EvC(X,W,b);
else
[P,~,~,~,~] = EvC(X,W,b,my_av, v_av);
end

correct = 0;
for i = 1:size(y,2)
    [~,maxindex(i)] = max(P(:,i));
    if y(i) == maxindex(i)
          correct = correct+1;
    end
end
acc = correct/size(y,2);
end

function [g] = BatchNormBP(g_init, S, my, v)

vb = diag(v + 10^(-6));
dv = 0;
du = 0;

for i=1:size(S,2)
     dv = dv + g_init(i,:)*(vb^(-3/2))*diag(S(:,i)-my);
     du = du + g_init(i,:)*vb^(-1/2);        
end

dv = dv*(-0.5);
du = du*(-1);

for j=1:size(S,2)
     g(j,:) = (g_init(j,:))*(vb^(-1/2)) + (2/size(S,2))*(dv)*(diag(S(:,j)-my))+du/size(S,2);
end

end


function [grad_W,grad_b] = CoG(X ,Y, W,b, lambda)

[P,S, my, v, s_hat , h ] = EvC(X,W,b);

grad_b = {};
grad_W = {};

g = -(Y-P)';

for i = length(W):-1:1

if i == 1
grad_b{i} = (sum(g,1)')/size(X,2);  
grad_W{i} = (g'*X')/size(X,2) + 2*lambda*W{i};

else

grad_b{i} = (sum(g,1)')/size(X,2);

grad_W{i} = (g'*h{i-1}')/size(X,2) + 2*lambda*W{i};

g = g*W{i};

g = g.*(s_hat{i-1}>0)';

g = BatchNormBP(g, S{i-1}, my{i-1}, v{i-1});

end

end
end

function [W, b, Jtrain, Jval,accTrain,accValid, my_av, v_av] = MBGDM(trainX, trainY,trainy, validX, validY,validy, n_batch, eta, n_epochs, W, b, lambda, rho, alph)
vW = {};
vb = {};
my_av = {};
v_av = {};

for a = 1:length(W)
    vW{a} = 0;
    vb{a} = 0;
end


for i = 1: n_epochs
    
    for j = 1:size(trainX,2)/n_batch
        j_start = (j-1)*n_batch+1;
        j_end = j*n_batch;
        Xbatch = trainX(:, j_start:j_end);
        
        Ybatch = trainY(:,j_start:j_end);
        
        [grad_W,grad_b] = CoG(Xbatch ,Ybatch, W,b, lambda);
    
    for m = 1:length(W)-1
        
        if j == 1
        [~, ~, my, v,~] = EvC(Xbatch, W, b);
        my_av{m} = my{m};
        v_av{m} = v{m};
        else
        [~, ~, my, v,~] = EvC(Xbatch, W, b);
        
        my_av{m} = alph*my_av{m}+(1-alph)*my{m};
        v_av{m} = alph*v_av{m}+(1-alph)*v{m};
        
        end
    end  
        
        for k = 1:length(W)
        vW{k} = rho*vW{k} + eta*grad_W{k};
        W{k} = W{k} - vW{k};
        
        vb{k} = rho*vb{k} + eta*grad_b{k};
        b{k} = b{k} - vb{k};
        
        end
    end
    eta = eta*0.95;
    Jtrain(i) = CoC(trainX,trainY,W,b,lambda,my_av, v_av);
    Jval(i) = CoC(validX,validY,W,b,lambda,my_av, v_av);
    accTrain(i) = CoA(trainX,trainy, W, b, my_av, v_av);
    accValid(i)= CoA(validX, validy, W, b, my_av, v_av);
    
end

end

