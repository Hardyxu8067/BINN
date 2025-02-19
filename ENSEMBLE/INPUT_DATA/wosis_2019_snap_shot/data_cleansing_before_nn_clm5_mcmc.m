clear,
clc;
%%
model_name = 'cesm2_clm5_cen_vr_v2';
time_domain = 'whole_time'; % 'whole_time', 'before_1985', 'after_1985', 'random_half'
% model_name = 'direct_nn';

is_median_scale = 0;

r2_threshold = 0.75;
gr_threshold = 1.05;
depth_threshold = 50;
%%
data_dir_output = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/OUTPUT_DATA/';
data_dir_input = '/Users/phoenix/Google_Drive/Tsinghua_Luo/Projects/DATAHUB/ENSEMBLE/INPUT_DATA/';

wosis_soc_info = ncread([data_dir_input, 'wosis_2019_snap_shot/soc_data_integrate_wosis_2019_snapshot_hugelius_mishra.nc'], 'data_soc_integrate'); % wosis SOC info
wosis_profile_info = ncread([data_dir_input, 'wosis_2019_snap_shot/soc_profile_wosis_2019_snapshot_hugelius_mishra.nc'], 'soc_profile_info'); % wosis profile info
profile_country = wosis_profile_info(:, 2);
profile_date = wosis_profile_info(:, 7);


if strcmp(model_name(1:7), 'clm_cen') == 1 || strcmp(model_name(1:5), 'cesm2') == 1
    para_mean = load([data_dir_output, 'mcmc_summary_', model_name, '/', model_name, '_para_mean.mat']);
    para_mean = para_mean.para_mean;
    stat_r2 = load([data_dir_output, 'mcmc_summary_', model_name, '/', model_name, '_stat_r2.mat']);
    stat_r2 = stat_r2.stat_r2;
    stat_r2 = max(stat_r2, [], 2);
    para_gr = load([data_dir_output, 'mcmc_summary_', model_name, '/', model_name, '_para_gr.mat']);
    para_gr = para_gr.para_gr;
    para_gr = mean(para_gr, 2);
    
    profile_env_info = load([data_dir_input, 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info.mat']);
    profile_env_info = profile_env_info.EnvInfo;
    grid_env_info = load([data_dir_input, 'data4nn/world_grid_envinfo_present.mat']);
    grid_env_info = grid_env_info.EnvInfo;
    
    profile_max_depth = profile_env_info(:, 73);
else
    profile_env_info = load([data_dir_input, 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info.mat']);
    profile_env_info = profile_env_info.EnvInfo;
    grid_env_info = load([data_dir_input, 'data4nn/world_grid_envinfo_present.mat']);
    grid_env_info = grid_env_info.EnvInfo;
end

%% find invalid profiles
invalid_profile = zeros(length(wosis_profile_info(:, 1)), 1);
for iprofile = 1:length(wosis_profile_info(:, 1))
    disp(['Processing profile: ', num2str(iprofile)]);
    profile_id = wosis_profile_info(iprofile, 1);
    profile_loc = find(wosis_soc_info(:, 1) == profile_id);
        
    layer_depth = wosis_soc_info(profile_loc, 5);
    layer_obs = wosis_soc_info(profile_loc, 7);
    
    % deal with nan values in obs or depth
    layer_valid_loc = find(isnan(layer_depth) == 0 & isnan(layer_obs) == 0);
    
    if isempty(layer_valid_loc) == 0
        layer_depth = layer_depth(layer_valid_loc);
        layer_obs = layer_obs(layer_valid_loc);
    else
        invalid_profile(iprofile) = 1;
        continue
    end

    [cor, pvalue] = corrcoef(layer_obs, layer_depth);
    
    if length(layer_obs) > 1 && cor(1, 2) > 0
        invalid_profile(iprofile) = 1;
    elseif length(layer_obs) < 2
        invalid_profile(iprofile) = 1;
    elseif find(layer_obs == max(layer_obs)) == length(layer_obs)
        invalid_profile(iprofile) = 1;
    elseif length(layer_obs) > 1 && layer_obs(1) <= layer_obs(end)
        invalid_profile(iprofile) = 1;
    end
    
end

eligible_loc_1 = find(invalid_profile == 0 & stat_r2 > 0 & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
    & profile_country ~= 68 & profile_country ~= 39);
eligible_loc_2 = find(invalid_profile == 0 & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
    & profile_country ~= 68 & profile_country ~= 39);
eligible_loc_3 = find(invalid_profile == 0 & stat_r2 > r2_threshold & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
            & profile_country ~= 68 & profile_country ~= 39);

eligible_loc_0 = find(invalid_profile == 0 & profile_max_depth > depth_threshold ...
            & profile_country ~= 68 & profile_country ~= 39);
        
save([data_dir_input, 'data4nn/eligible_profile_loc_0_', model_name, '_', time_domain, '.mat'], 'eligible_loc_0');
save([data_dir_input, 'data4nn/eligible_profile_loc_1_', model_name, '_', time_domain, '.mat'], 'eligible_loc_1');
save([data_dir_input, 'data4nn/eligible_profile_loc_2_', model_name, '_', time_domain, '.mat'], 'eligible_loc_2');
save([data_dir_input, 'data4nn/eligible_profile_loc_3_', model_name, '_', time_domain, '.mat'], 'eligible_loc_3');

%% cleans the data
if strcmp(model_name(1:7), 'clm_cen') == 1 || strcmp(model_name(1:5), 'cesm2') == 1
    profile_env_info(:, end) = stat_r2;
    % eliminate profiles r2 < 0
    if strcmp(time_domain, 'whole_time') == 1
        valid_loc = find(invalid_profile == 0 & stat_r2 > r2_threshold & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
            & profile_country ~= 68 & profile_country ~= 39);
    elseif strcmp(time_domain, 'before_1985') == 1
        valid_loc = find(invalid_profile == 0 & stat_r2 > r2_threshold & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
            & profile_country ~= 68 & profile_country ~= 39 & profile_date <= 1985);
    elseif strcmp(time_domain, 'after_1985') == 1
        valid_loc = find(invalid_profile == 0 & stat_r2 > r2_threshold & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
            & profile_country ~= 68 & profile_country ~= 39 & profile_date > 1985);
    elseif strcmp(time_domain, 'random_half_1') == 1
        valid_loc_middle = find(invalid_profile == 0 & stat_r2 > r2_threshold & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
            & profile_country ~= 68 & profile_country ~= 39);
        valid_loc = sort(datasample(valid_loc_middle, round(length(valid_loc_middle)/2), 'Replace', false));
    elseif strcmp(time_domain, 'random_half_2') == 1
        valid_loc_middle = find(invalid_profile == 0 & stat_r2 > r2_threshold & profile_max_depth > depth_threshold & para_gr < gr_threshold ...
            & profile_country ~= 68 & profile_country ~= 39);
        first_half = load([data_dir_input, 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info_', model_name, '_random_half_1_maxmin_scaled.mat'], 'profile_env_info');
        first_half = first_half.profile_env_info(:, 1);
        valid_loc = setdiff(valid_loc_middle, first_half);
    end
    profile_env_info = profile_env_info(valid_loc, :);
    para_mean = para_mean(valid_loc, :);
end

profile_env_info_origin = profile_env_info;
%% normalize env info

col_max_min = nan(length(profile_env_info(1, :)), 2);
for icol = 4:length(profile_env_info(1, :)) - 1 % 4:end-1 is corresponding to the variable names
    col_count = icol - 3;
    if icol == 73
        col_median = median(profile_env_info(:, icol), 'omitnan');
        col_max = max(profile_env_info(:, icol), [], 'omitnan');
        col_min = min(profile_env_info(:, icol), [], 'omitnan');
    else
        col_median = median(grid_env_info(:, col_count), 'omitnan');
        col_max = max(grid_env_info(:, col_count), [], 'omitnan');
        col_min = min(grid_env_info(:, col_count), [], 'omitnan');
        if col_median == col_max || col_median == col_min
            col_median = mean(profile_env_info(:, col_count), 'omitnan');
        end
    end
    
    col_max_min(icol, :) = [col_min, col_max];
    
    if is_median_scale == 1
        % scale profile env info
        loc_above_median = find(profile_env_info(:, icol) >= col_median);
        loc_below_median = find(profile_env_info(:, icol) < col_median);
        
        profile_env_info(loc_above_median, icol) = ...
            (profile_env_info(loc_above_median, icol) - col_median)/(col_max - col_median)*0.5;
        profile_env_info(loc_below_median, icol) = ...
            (profile_env_info(loc_below_median, icol) - col_median)/(col_median - col_min)*0.5;
        
        % scale grid env info
        loc_above_median = find(grid_env_info(:, col_count) >= col_median);
        loc_below_median = find(grid_env_info(:, col_count) < col_median);
        
        grid_env_info(loc_above_median, col_count) = ...
            (grid_env_info(loc_above_median, col_count) - col_median)/(col_max - col_median)*0.5;
        grid_env_info(loc_below_median, col_count) = ...
            (grid_env_info(loc_below_median, col_count) - col_median)/(col_median - col_min)*0.5;
    else
        % scale profile env info
        profile_env_info(:, icol) = ...
            (profile_env_info(:, icol) - col_min)/(col_max - col_min);
        
        % scale grid env info
        grid_env_info(:, col_count) = ...
            (grid_env_info(:, col_count) - col_min)/(col_max - col_min);
    end
    
end

if is_median_scale == 1
    if strcmp(model_name(1:7), 'clm_cen') == 1 || strcmp(model_name(1:5), 'cesm2') == 1
        save([data_dir_output, 'mcmc_summary_', model_name, '/', model_name, '_', time_domain, '_para_mean_cleansed.mat'], 'para_mean');
        save([data_dir_input, 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info_', model_name, '_', time_domain, '_median_scaled.mat'], 'profile_env_info');
        save([data_dir_input, 'data4nn/world_grid_envinfo_present_', model_name, '_', time_domain, '_median_scaled.mat'], 'grid_env_info');
    else
        save([data_dir_input, 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info_', model_name, '_median_scaled.mat'], 'profile_env_info');
        save([data_dir_input, 'data4nn/world_grid_envinfo_present_', model_name, '_median_scaled.mat'], 'grid_env_info');
    end
else
    if strcmp(model_name(1:7), 'clm_cen') == 1 || strcmp(model_name(1:5), 'cesm2') == 1
        save([data_dir_output, 'mcmc_summary_', model_name, '/', model_name, '_', time_domain, '_para_mean_cleansed.mat'], 'para_mean');
        save([data_dir_input, 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info_', model_name, '_', time_domain, '_maxmin_scaled.mat'], 'profile_env_info');
        save([data_dir_input, 'data4nn/world_grid_envinfo_present_', model_name, '_', time_domain, '_maxmin_scaled.mat'], 'grid_env_info');
        save([data_dir_input, 'data4nn/world_grid_envinfo_present_', model_name, '_', time_domain, '_col_max_min.mat'], 'col_max_min');
    else
        save([data_dir_input, 'wosis_2019_snap_shot/wosis_2019_snapshot_hugelius_mishra_env_info_', model_name, '_maxmin_scaled.mat'], 'profile_env_info');
        save([data_dir_input, 'data4nn/world_grid_envinfo_present_', model_name, '_maxmin_scaled.mat'], 'grid_env_info');
    end
end


disp('Program finished');