function [carbon_input, cpools, cpools_layer, soc_layer, soc_stock, site_input_ratio] = ...
    fun_var_decom(nbedrock, sand_vector, npp_mean, ...
    input_vector_cwd, input_vector_litter1, input_vector_litter2, input_vector_litter3, ...
    altmax_current_profile, altmax_lastyear_profile, soil_temp_profile, soil_water_profile, xio, xin)
% define global parameters
% global diffus adv cryo q10 fq10 maxpsi minpsi efolding ...
%     tau4cwd tau4l1 tau4l2 tau4l3 tau4s1 tau4s2 tau4s3 fl1s1 fl2s1 fl3s2 fs2s1 fs3s1 ...
%     fs1s2 fs1s3 fs2s3 fcwdl2 fcwdl3 ins beta4cwd beta4l1 beta4l2 beta4l3 dmax p4cwd ...
%     p4ml p4cl p4ll
month_num = 12;
global kelvin_to_celsius
kelvin_to_celsius = 273.15;

global normalize_q10_to_century_tfunc
normalize_q10_to_century_tfunc = false;

global use_vertsoilc npool npool_vr n_soil_layer days_per_year secspday
use_vertsoilc = 1 ; % whether or not use vertical maxing part
npool = 7;  % number of pools if no vertical
npool_vr = 140; % number of pools if vertical
n_soil_layer = 20;  % number of soil layers
days_per_year = 365;
secspday = 24*60*60;
% dt = secspday*30;

global max_altdepth_cryoturbation max_depth_cryoturb
max_altdepth_cryoturbation = 2;
max_depth_cryoturb = 3;

global dz dz_node zisoi zsoi
% width between two interfaces
dz = [2.000000000000000E-002, 4.000000000000000E-002, 6.000000000000000E-002, ...
    8.000000000000000E-002, 0.120000000000000, 0.160000000000000, ...
    0.200000000000000, 0.240000000000000, 0.280000000000000, ...
    0.320000000000000, 0.360000000000000, 0.400000000000000, ...
    0.440000000000000, 0.540000000000000, 0.640000000000000, ...
    0.740000000000000, 0.840000000000000, 0.940000000000000, ...
    1.04000000000000, 1.14000000000000, 2.39000000000000, ...
    4.67553390593274, 7.63519052838329, 11.1400000000000, ...
    15.1154248593737]';

% depth of the interface
zisoi = [2.000000000000000E-002, 6.000000000000000E-002, ...
    0.120000000000000, 0.200000000000000, 0.320000000000000, ...
    0.480000000000000, 0.680000000000000, 0.920000000000000, ...
    1.20000000000000, 1.52000000000000, 1.88000000000000, ...
    2.28000000000000, 2.72000000000000, 3.26000000000000, ...
    3.90000000000000, 4.64000000000000, 5.48000000000000, ...
    6.42000000000000, 7.46000000000000, 8.60000000000000, ...
    10.9900000000000, 15.6655339059327, 23.3007244343160, ...
    34.4407244343160, 49.5561492936897]';

% depth of the node
zsoi = [1.000000000000000E-002, 4.000000000000000E-002, 9.000000000000000E-002, ...
    0.160000000000000, 0.260000000000000, 0.400000000000000, ...
    0.580000000000000, 0.800000000000000, 1.06000000000000, ...
    1.36000000000000, 1.70000000000000, 2.08000000000000, ...
    2.50000000000000, 2.99000000000000, 3.58000000000000, ...
    4.27000000000000, 5.06000000000000, 5.95000000000000, ...
    6.94000000000000, 8.03000000000000, 9.79500000000000, ...
    13.3277669529664, 19.4831291701244, 28.8707244343160, ...
    41.9984368640029]';

% depth between two node
dz_node = zsoi - [0; zsoi(1:end-1)];

% default parameter values
% default_para_dir = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/INPUT_DATA/cesm2_parameters/clm5_params.c190829.nc';
default_para_dir = '/GFPS8p/cess11/taof/datahub/ensemble/input_data/cesm2_parameters/clm5_params.c190829.nc';

% define parameter names
bio = ncread(default_para_dir, 'som_diffus')*3600*24*365; % diffusion (bioturbation?) (m2/yr)
cryo = ncread(default_para_dir, 'cryoturb_diffusion_k')*3600*24*365; % exp(kp(2)*(-7 - (-12)) + (-12)); % cryoturbation (m2/yr)
q10 = ncread(default_para_dir, 'q10_hr'); % Q10 (unitless)
fq10 = ncread(default_para_dir, 'froz_q10'); % Q10 when forzen (unitless)
efolding = ncread(default_para_dir, 'decomp_depth_efolding'); % parameters used in vertical discretization of carbon inputs (metre)

tau4cwd = ncread(default_para_dir, 'tau_cwd'); % kp(5)*(10 - 1) + 1; % turnover time of CWD (yr) 4.1 in technote
tau4l1 = ncread(default_para_dir, 'tau_l1'); % kp(6)*(0.2 - 0) + 0; % tau for metabolic litter (yr) 0.066 in technote
tau4l2 = ncread(default_para_dir, 'tau_l2_l3'); %kp(7)*(1 - 0.2) + 0.2; % tau for cellulose litter (yr) 0.25 in technote
tau4l3 = tau4l2; % tau for lignin litter (yr)

tau4s1 = ncread(default_para_dir, 'tau_s1'); % tau for fast SOC (yr)
tau4s2 = ncread(default_para_dir, 'tau_s2'); % tau for slow SOC (yr)
tau4s3 = ncread(default_para_dir, 'tau_s3'); % tau for passive SOC (yr)

fl1s1 = 1 - ncread(default_para_dir, 'rf_l1s1_bgc'); % 1 - 0.55;
fl2s1 = 1 - ncread(default_para_dir, 'rf_l2s1_bgc'); % 1 - 0.5;
fl3s2 = 1 - ncread(default_para_dir, 'rf_l3s2_bgc'); % 1 - 0.5;
fs1s2 = nan;
fs1s3 = nan;
fs2s1 = (0.42/0.45)*(1 - ncread(default_para_dir, 'rf_s2s1_bgc')); % (0.42/0.45)*(1 - 0.55);
fs2s3 = (0.03/0.45)*(1 - ncread(default_para_dir, 'rf_s2s3_bgc')); % (0.03/0.45)*(1 - 0.55);
fs3s1 = 1 - ncread(default_para_dir, 'rf_s3s1_bgc'); % 1 - 0.55;
fcwdl2 = 1 - ncread(default_para_dir, 'cwd_flig'); % 0.76;

% maximum and minimum water potential (MPa)
maxpsi= ncread(default_para_dir, 'maxpsi_hr'); % -0.002;

minpsi= ncread(default_para_dir, 'minpsi_hr'); % -2; % minimum water potential (MPa)

adv = 0; % parameter for advection (m/yr)

%% Environmental Scalar (Xi)
xit = nan(n_soil_layer, month_num);
xiw = nan(n_soil_layer, month_num);

for imonth = 1:month_num
    % temperature related function xit
    % calculate rate constant scalar for soil temperature
    % assuming that the base rate constants are assigned for non-moisture
    % limiting conditions at 25 C.
    for ilayer = 1 : n_soil_layer
        if soil_temp_profile(ilayer, imonth) >= 0 + kelvin_to_celsius
            xit(ilayer, imonth) = q10^((soil_temp_profile(ilayer, imonth) - (kelvin_to_celsius + 25))/10);
        else
            xit(ilayer, imonth) = q10^((273.15 - 298.15)/10)*(fq10^((soil_temp_profile(ilayer, imonth) - (0 + kelvin_to_celsius))/10));
        end
    end
    
    catanf_30 = catanf(30);
    normalization_tref = 15;
    if normalize_q10_to_century_tfunc == true
        % scale all decomposition rates by a constant to compensate for offset between original CENTURY temp func and Q10
        normalization_factor = (catanf(normalization_tref)/catanf_30) / (q10^((normalization_tref-25)/10));
        xit = xit*normalization_factor;
    end
    
    % water related function xiw
    % calculate the rate constant scalar for soil water content.
    % Uses the log relationship with water potential given in
    % Andren, O., and K. Paustian, 1987. Barley straw decomposition in the field:
    % a comparison of models. Ecology, 68(5):1190-1200.
    % and supported by data in
    % Orchard, V.A., and F.J. Cook, 1983. Relationship between soil respiration
    % and soil moisture. Soil Biol. Biochem., 15(4):447-453.
    
    for ilayer = 1 : n_soil_layer
        psi = min(maxpsi, soil_water_profile(ilayer, imonth));
        
        if psi > minpsi
            xiw(ilayer, imonth) = (log(minpsi/psi))/(log(minpsi/maxpsi));
        else
            xiw(ilayer, imonth) = 0;
        end
        xiw(ilayer, imonth) = max(xiw(ilayer, imonth), 0.05);
    end
end

% xit = soil_temp_profile;

xiw = soil_water_profile;

%% Triangle Matrix, A Matrix and K Matrix
sand_vector = mean(sand_vector, 2, 'omitnan');
% Allocation matrix
a_ma = a_matrix(fl1s1, fl2s1, fl3s2, fs1s2, fs1s3, fs2s1, fs2s3, fs3s1, fcwdl2, sand_vector);

kk_ma_middle = nan(npool_vr, npool_vr, month_num);
tri_ma_middle = nan(npool_vr, npool_vr, month_num);
for imonth = 1:month_num
    % decomposition matrix
    monthly_xit = xit(:, imonth);
    monthly_xiw = xiw(:, imonth);
    monthly_xio = xio(:, imonth);
    monthly_xin = xin(:, imonth);
    kk_ma_middle(:, :, imonth) = kk_matrix(monthly_xit, monthly_xiw, monthly_xio, monthly_xin, efolding, tau4cwd, tau4l1, tau4l2, tau4l3, tau4s1, tau4s2, tau4s3);
    % tri matrix
    monthly_nbedrock = nbedrock(imonth);
    monthly_altmax_current_profile = altmax_current_profile(imonth);
    monthly_altmax_lastyear_profile = altmax_lastyear_profile(imonth);
    tri_ma_middle(:, :, imonth) = tri_matrix(monthly_nbedrock, monthly_altmax_current_profile, monthly_altmax_lastyear_profile, bio, adv, cryo);
end
tri_ma = mean(tri_ma_middle, 3, 'omitnan');
kk_ma = mean(kk_ma_middle, 3, 'omitnan');

matrix_in=nan(140,1);

input_vector_cwd = sum(input_vector_cwd, 2, 'omitnan');
input_vector_litter1 = sum(input_vector_litter1, 2, 'omitnan');
input_vector_litter2 = sum(input_vector_litter2, 2, 'omitnan');
input_vector_litter3 = sum(input_vector_litter3, 2, 'omitnan');

matrix_in(1:20,1) = input_vector_cwd./days_per_year; % litter input gc/m3/day
matrix_in(21:40,1) = input_vector_litter1./days_per_year;
matrix_in(41:60,1) = input_vector_litter2./days_per_year;
matrix_in(61:80,1) = input_vector_litter3./days_per_year;
matrix_in(81:140,1) = 0;

carbon_input = sum((input_vector_cwd + input_vector_litter1 + input_vector_litter2 + input_vector_litter3).*dz(1:n_soil_layer));

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]';
site_input_ratio = carbon_input/(sum(npp_mean.*days_in_month*24*3600));

cpools = (a_ma*kk_ma-tri_ma)\(-matrix_in);
cpools(cpools < 0) = 0;
cpools_layer = sum([cpools(1:20), cpools(21:40), cpools(41:60), cpools(61:80), cpools(81:100), cpools(101:120), cpools(121:140)], 2);
soc_layer = sum([cpools(81:100), cpools(101:120), cpools(121:140)], 2);
soc_stock = sum(soc_layer.*dz(1:n_soil_layer));

end

% ----- CENTURY T response function
function catanf_results = catanf(t1)
catanf_results = 11.75 +(29.7 / pi) * atan( pi * 0.031  * ( t1 - 15.4 ));
end

