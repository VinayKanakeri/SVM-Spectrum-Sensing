function SVM = LSVM(train, test)

scoreModel = fitcsvm(train.X,train.Y,'KernelFunction', 'linear');
SVM.model = fitPosterior(scoreModel); % Transforms score to posterior probability
[predictClass,SVM.P] = predict(SVM.model,test.X);
error = sum(predictClass ~= test.Y, 'all')/length(test.Y);
SVM.positiveClass = 2;
SVM.name = 'LSVM';
SVM.error = error;