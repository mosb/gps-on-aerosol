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

%load aeronet_gp_4

for ii=1:2
    
    

    if ii==1
        % Terra
        fn='intersection_MODIS_5_MOD04_L2.dat';        
        %covvy = gp_terra;
    else
        % Aqua
        fn='intersection_MODIS_5_MYD04_L2.dat';     
        %covvy = gp_aqua;
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
    
%     figure
%     hold on
%     for i = 1:size(p,2)
%         range_i = std(p_train(:,i));
%         mean_i = mean(p_train(:,i));
%         plot(i,(p_test(:,i)-mean_i)/range_i,'bo')
%         plot(i,(p_train(:,i)-mean_i)/range_i,'r+')
%     end
    

    
    % set the time available for optimisation
    params.max_fn_evals = 100000;
    
    covvy = set_covvy('matern', 'planar', [], p_train, t_train, 1);

    % training
    [dummy,dummy2,covvy,closestInd] = predict_ML([],p_train,t_train,covvy,params);
    
    % testing
    [t_mean,t_sd]=predict_ML(p_test,p_train,[],covvy);
    
    if ii==1
        % Terra
        gp_terra=covvy;
        closestInd_terra = closestInd;
        t_mean_terra = t_mean;
        t_sd_terra = t_sd;
        t_terra=t_test;
        p_terra=p_test;
        p_train_terra = p_train;
        t_train_terra = t_train;
    else
        % Aqua
        gp_aqua=covvy;
        closestInd_aqua = closestInd;
        MOptical_Depth_Land_And_Ocean_aqua=MOptical_Depth_Land_And_Ocean(iwant);
        t_mean_aqua = t_mean;
        t_sd_aqua = t_sd;
        t_aqua=t_test;
        p_aqua=p_test;
        p_train_aqua = p_train;
        t_train_aqua = t_train;
    end

end



%--------------------------------------------------------------------------
max_t=max([max(t_terra) max(t_aqua)])
tp=[0 max_t];

%--------------------------------------------------------------------------
% Calculate the correlation coefficients

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% uncorrected satellite vs aeronet
[R,P]=corrcoef(t_terra,p_terra(:,end));
rr_terra=R(2,1);
pp_terra=P(2,1);    

%--------------------------------------------------------------------------
% GP corrected satellite vs aeronet
[R,P]=corrcoef(t_terra,t_mean_terra);
rr_gp_terra=R(2,1);
pp_gp_terra=P(2,1);    

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% uncorrected satellite vs aeronet
[R,P]=corrcoef(t_aqua,p_aqua(:,end));
rr_aqua=R(2,1);
pp_aqua=P(2,1);    

%--------------------------------------------------------------------------
% GP corrected satellite vs aeronet
[R,P]=corrcoef(t_aqua,t_mean_aqua);
rr_gp_aqua=R(2,1);
pp_gp_aqua=P(2,1);    

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Draw a scatter plot 
figure('Position',[left, bottom, width, height])

% GP Aeronet - corrected MODIS Terra comparison
plot(t_terra,t_mean_terra,'+r')   
%correlation_plot(t_mean_terra, t_sd_terra, t_terra, params)

% Raw Aeronet - MODIS Terra comparison
plot(t_terra,p_terra(:,end),'sg')    


% Raw Aeronet - MODIS aqua comparison
plot(t_aqua,p_aqua(:,end),'oc')    

% GP Aeronet - corrected MODIS aqua comparison
plot(t_aqua,t_mean_aqua,'.k')    
%correlation_plot(t_mean_aqua, t_sd_aqua, t_aqua, params)
hold off    

xlabel('AERONET AOD','FontSize',22)
ylabel('MODIS (Terra/Aqua)','FontSize',22)
title(['MODIS AOD Comparison at 550 nm '],'FontSize',22)
grid on
legend('1:1',['Terra, R=' num2str(rr_terra,2)],['Aqua, R=' num2str(rr_aqua,2)],['Terra GP, R=' num2str(rr_gp_terra,2)],['Aqua GP, R=' num2str(rr_gp_aqua,2)],'Location','NorthWest')
xlim([0 max_t])
ylim([0 max_t])
set(gca,'FontSize',16);
fn=['GP-scatter'];
[fnpng,fnps]=wrplotepspng(fn);

%--------------------------------------------------------------------------
left=left+width;
if left>3*width
    left=1;
    bottom=bottom+height;
end

%--------------------------------------------------------------------------

%MOptical_Depth_Land_And_Ocean is the most important input
input_importances_terra = input_importance(gp_terra, p_train_terra, closestInd_terra)'
input_importances_aqua = input_importance(gp_aqua, p_train_aqua, closestInd_aqua)'

input_importances_terra_wm = input_importance_wm(gp_terra, p_terra, p_train_terra, closestInd_terra)'
input_importances_aqua_wm = input_importance_wm(gp_aqua, p_aqua, p_train_aqua, closestInd_aqua)'


clear covvy separation Solar_Zenith_Angle TheDate VI MScattering_Angle MSensor_Azimuth MSensor_Zenith MSLException MSolar_Azimuth MSolar_Zenith
save aeronet_gp_6 -v7.3
% load aeronet_gp tp t_terra p_terra t_aqua p_aqua  max_t rr_gp_terra rr_gp_aqua rr_aqua rr_terra t_aqua t_mean_aqua t_terra t_mean_terra