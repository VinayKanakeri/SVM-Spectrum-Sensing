function model = predict(test, model)
Pfa_target = 2.5e-3:2.5e-3:1;

X = test.X;
Y = test.Y;
len = length(Pfa_target);

model.Pd = zeros(len,1);
model.Pfa = zeros(len,1);
for i=1:length(Pfa_target)
    alpha = 1-Pfa_target(i); 
    status_svm = model.P(:,model.positiveClass)>=alpha;
    detected_svm = Y & status_svm;
    falseAlarm_svm = logical(status_svm - detected_svm);
    model.Pd(i) = sum(detected_svm)/sum(Y);
    model.Pfa(i) = sum(falseAlarm_svm)/(length(Y)-sum(Y));
end
end