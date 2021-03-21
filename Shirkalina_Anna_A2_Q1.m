clear variables;
clc;
clf;

%parameters and initialization 
eta = 0.0005;
%y = shirkalina_a_create_mapnet;([%TODO]) activation = tanh;
epoch = 10;
    
net = shirkalina_a_create_mapnet();

while epoch > 0
    load 'C:\Users\Anya\OneDrive - University of Toronto\School\Year 3\PSL432\MNIST.mat';  % substitute the appropriate folder on your computer
    shuffle = randperm(size(TRAIN_images, 1));
    TRAIN_images = TRAIN_images(shuffle, :);
    TRAIN_answers = TRAIN_answers(shuffle, :);
    TRAIN_labels = TRAIN_labels(shuffle, :);
    
    for i = 1:100:60000
        images = TRAIN_images(i:(99+i),:)'; %transposing the images, bc i want each image to be a colunm vector answer
        y_star = TRAIN_labels(i:(99+i),:)';
        net = shirkalina_a_forward_rectanh(net, images); 
        e = net.y{3} - y_star;
        %%%For loop to calculate the loss, because the e is actually a
        %%%matrix we have to compute the loss seperatly for each column,
        %%%not strictly necessary for backprop-but it's here none the less
        %L = zeros(1, 100);
        %for j=1:1:100
        % L(1, j) = 0.5 * e(:,j)'*e(:, j);
        %end
        %L = mean(L, 1); %taking the averge of all of the losses
        net = shirkalina_a_backprop_rectanh(net, e);
        net = adam(net, net.dL_dW, net.dL_db, eta, 0.9, 0.999);   
    end
    %if epoch==10 || epoch==1
        shuffle = randperm(size(TEST_images, 1));
        test_images = TEST_images(shuffle, :)'; %transposing the images, bc i want each image to be a column vector answer
        test_ans = TEST_labels(shuffle, :)';
        test_net = shirkalina_a_create_mapnet();
        test_net = shirkalina_a_forward_rectanh(test_net, test_images);
        output = test_net.y{3} == max(test_net.y{3});
        result = sum(output ~= test_ans, 'all') / 2
   % end
        
    
    epoch = epoch - 1
end   