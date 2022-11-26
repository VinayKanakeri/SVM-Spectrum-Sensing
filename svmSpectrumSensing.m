clear
close all
addpath('dataset');

train = load('snr_10_ch1_svm_sample_5.mat').dataset;
test = load('snr_10_ch1_svm_sample_5_test.mat').dataset;
meanSNRdB = mean(test.snr(test.snr ~= -Inf)) - 10*log10(15360); % Adjustment for normaliation missed while generating the dataset

epochs = size(train, 2);
linearSVM = struct();
gaussianSVM = struct();

Pfa_target = 2.5e-3:2.5e-3:1;

for ep=1:epochs
    linearSVM(ep).model = LSVM(train(ep), test);
    gaussianSVM(ep).model = GSVM(train(ep), test);
    linearSVM(ep).model = predict(test, linearSVM(ep).model);
    gaussianSVM(ep).model = predict(test, gaussianSVM(ep).model);
end

linearSVMPd = zeros(length(Pfa_target), 1);
linearSVMPfa = zeros(length(Pfa_target), 1);
gaussianSVMPd = zeros(length(Pfa_target), 1);
gaussianSVMPfa = zeros(length(Pfa_target), 1);
errorLSVM = 0;
errorGSVM = 0;
for ep=1:epochs
    linearSVMPd = linearSVMPd + linearSVM(ep).model.Pd;
    linearSVMPfa = linearSVMPfa + linearSVM(ep).model.Pfa;
    gaussianSVMPd = gaussianSVMPd + gaussianSVM(ep).model.Pd;
    gaussianSVMPfa = gaussianSVMPfa + gaussianSVM(ep).model.Pfa;
    errorLSVM = linearSVM(ep).model.error;
    errorGSVM = gaussianSVM(ep).model.error;
end

linearSVMPd = linearSVMPd/epochs;
linearSVMPfa = linearSVMPfa/epochs;
gaussianSVMPd = gaussianSVMPd/epochs;
gaussianSVMPfa = gaussianSVMPfa/epochs;
errorGSVM = errorGSVM/epochs;
errorLSVM = errorLSVM/epochs;

fg = figure('Name','ROC Plot','NumberTitle','off');
figure(fg)

plot(linearSVMPfa, linearSVMPd)
hold on
plot(gaussianSVMPfa, gaussianSVMPd)
grid on
legend(sprintf('Linear SVM, Error = %f', errorLSVM), sprintf('Gaussian SVM, Error = %f', errorGSVM))
ylabel('$P_{d}$', 'Interpreter', 'latex');
xlabel('$P_{fa}$', 'Interpreter', 'latex');
title(sprintf('SNR = %f', meanSNRdB));


