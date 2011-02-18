clear;clc;close all

%--------------------------------------------------------------------------
format='%s';
for i=1:14
    format=[format ' %f'];
end
%--------------------------------------------------------------------------

width=600;
height=400;
left=1;
bottom=1;

%--------------------------------------------------------------------------

for ii=1:2

    if ii==1
        % Terra
        fn='intersection_MODIS_5_MOD04_L2.dat';        
    else
        % Aqua
        fn='intersection_MODIS_5_MYD04_L2.dat';                
    end
    disp(fn)
    [TheDate,Latitude,Longitude,Solar_Zenith_Angle,VI,Elevation,MSolar_Zenith,MSolar_Azimuth,MSensor_Zenith,MSensor_Azimuth,MScattering_Angle,MReflectance,separation,MOptical_Depth_Land_And_Ocean,AOT_550]=textread(fn,format,'delimiter',',');
    disp(length(AOT_550))

    %--------------------------------------------------------------------------
    dsza=(MSolar_Zenith-Solar_Zenith_Angle);
    iwant=find(separation<0.3 & abs(dsza)<0.3);
    iwant=find(separation<0.1 & abs(dsza)<0.1);
    iwant=find(separation<0.2 & abs(dsza)<0.2);
    disp(length(iwant))

    %p=[Solar_Zenith_Angle(iwant) VI(iwant) Elevation(iwant) MSolar_Zenith(iwant) MSolar_Azimuth(iwant) MSensor_Zenith(iwant) MSensor_Azimuth(iwant) MScattering_Angle(iwant) MReflectance(iwant) separation(iwant) MOptical_Depth_Land_And_Ocean(iwant)];
    %p=[VI(iwant) Elevation(iwant) MSolar_Zenith(iwant) MSolar_Azimuth(iwant) MSensor_Zenith(iwant) MSensor_Azimuth(iwant) MScattering_Angle(iwant) MReflectance(iwant) MOptical_Depth_Land_And_Ocean(iwant)];
    p=[MSolar_Zenith(iwant) MSolar_Azimuth(iwant) MSensor_Zenith(iwant) MSensor_Azimuth(iwant) MScattering_Angle(iwant) MReflectance(iwant) MOptical_Depth_Land_And_Ocean(iwant)];
    %p=[VI(iwant) MOptical_Depth_Land_And_Ocean(iwant)];
    t=[AOT_550(iwant) ];
    
    
    training_inds = 1:2:size(t,1);
    p_train = p(training_inds,:);
    t_train = t(training_inds,:);
    
    testing_inds = 2:2:size(t,1);
    p_test = p(testing_inds,:);
    t_test = t(testing_inds,:);

%   Commented out the NN fit.    
    % p=p';
    % t=t';
    % 
    % numHiddenNeurons = 40;  % Adjust as desired
    % net = create_fit_net(p,t,numHiddenNeurons);

%     n=0.00053103;
%     g=36.273;
%     c=36.273;
%     r=-0.0066558;
%     d=0.43607;


    %     options:
    %     -s svm_type : set type of SVM (default 0)
    %         0 -- C-SVC
    %         1 -- nu-SVC
    %         2 -- one-class SVM
    %         3 -- epsilon-SVR
    %         4 -- nu-SVR
    %     -t kernel_type : set type of kernel function (default 2)
    %         0 -- linear: u'*v
    %         1 -- polynomial: (gamma*u'*v + coef0)^degree
    %         2 -- radial basis function: exp(-gamma*|u-v|^2)
    %         3 -- sigmoid: tanh(gamma*u'*v + coef0)
    %     -d degree : set degree in kernel function (default 3)
    %     -g gamma : set gamma in kernel function (default 1/k)
    %     -r coef0 : set coef0 in kernel function (default 0)
    %     -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    %     -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    %     -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
    %     -m cachesize : set cache memory size in MB (default 100)
    %     -e epsilon : set tolerance of termination criterion (default 0.001)
    %     -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
    %     -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
    %     -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
    % 
    %     The k in the -g option means the number of attributes in the input data.
    % 
    %     option -v randomly splits the data into n parts and calculates cross
    %     validation accuracy/mean squared error on them.
    disp('training SVM ...')
    %     n=0.0005;
%     g=40;
%     c=40;
%     r=0;
%     d=0.5;
%    libsvm_options=['-s 3 -t 2 -p ' num2str(n) ' -g ' num2str(g) ' -c ' num2str(c) ' -d ' num2str(d)];
    
    libsvm_options=['-s 3 -t 2'];
    tic
    model = svmtrain(t_train, p_train, libsvm_options);
    toc

    tic
    [SVM_AOT_550, accuracy, dec_values] = svmpredict(t_test, p_test, model); % test the training data
    toc
    
    if ii==1
        % Terra
        model_terra=model;
        MOptical_Depth_Land_And_Ocean_terra=MOptical_Depth_Land_And_Ocean(iwant);
        SVM_terra=SVM_AOT_550;
        t_terra=t_test;
        p_terra=p_test;
    else
        % Aqua
        model_aqua=model;
        MOptical_Depth_Land_And_Ocean_aqua=MOptical_Depth_Land_And_Ocean(iwant);
        SVM_aqua=SVM_AOT_550;
        t_aqua=t_test;
        p_aqua=p_test;
    end

end

%--------------------------------------------------------------------------
max_t=max([max(t_terra) max(t_aqua)])
tp=[0 max_t];

%--------------------------------------------------------------------------
% Calculate the correlation coefficients

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
[R,P]=corrcoef(t_terra,p_terra(:,end));
rr_terra=R(2,1);
pp_terra=P(2,1);    

%--------------------------------------------------------------------------
[R,P]=corrcoef(t_terra,SVM_terra);
rr_svm_terra=R(2,1);
pp_svm_terra=P(2,1);    

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
[R,P]=corrcoef(t_aqua,p_aqua(:,end));
rr_aqua=R(2,1);
pp_aqua=P(2,1);    

%--------------------------------------------------------------------------
[R,P]=corrcoef(t_aqua,SVM_aqua);
rr_svm_aqua=R(2,1);
pp_svm_aqua=P(2,1);    

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Draw a scatter plot 
figure('Position',[left, bottom, width, height])

% Perfect 1:1 line
plot(tp,tp,'-b','LineWidth',2)    
hold on

% Raw Aeronet - MODIS Terra comparison
plot(t_terra,p_terra(:,end),'sg')    

% Raw Aeronet - MODIS aqua comparison
plot(t_aqua,p_aqua(:,end),'oc')    

% SVM Aeronet - corrected MODIS Terra comparison
plot(t_terra,SVM_terra,'+r')    

% SVM Aeronet - corrected MODIS aqua comparison
plot(t_aqua,SVM_aqua,'.k')    
hold off    

xlabel('AERONET AOD','FontSize',22)
ylabel('MODIS (Terra/Aqua)','FontSize',22)
title(['MODIS AOD Comparison at 550 nm '],'FontSize',22)
grid on
legend('1:1',['Terra, R=' num2str(rr_terra,2)],['Aqua, R=' num2str(rr_aqua,2)],['Terra SVM, R=' num2str(rr_svm_terra,2)],['Aqua SVM, R=' num2str(rr_svm_aqua,2)],'Location','NorthWest')
xlim([0 max_t])
ylim([0 max_t])
set(gca,'FontSize',16);
fn=['svm-scatter'];
[fnpng,fnps]=wrplotepspng(fn);

%--------------------------------------------------------------------------
left=left+width;
if left>3*width
    left=1;
    bottom=bottom+height;
end

%--------------------------------------------------------------------------

clear separation Solar_Zenith_Angle TheDate VI MScattering_Angle MSensor_Azimuth MSensor_Zenith MSLException MSolar_Azimuth MSolar_Zenith
save aeronet_svm_6 -v7.3
