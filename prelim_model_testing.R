library(tidyverse)
library(here)
library(catboost)

############################################### Description ############################################### 
#' preliminary file to run before cv training and evaluation
#' to quickly get a sense if new features are beneficial and whether they're worth including
#' basically just runs cv on two different datasets to see if adding features improves cv or not


source(here("helper.R"))
close_features = read.csv(here("data", "closest_dir_dist_features.csv")) %>%
  select(game_player_play_id, frame_id, starts_with("closest_"))
data_mod = read.csv(file = here("data", "data_mod_no_close_player.csv")) %>%
  mutate(across(where(is.character), as.factor)) #for catboost
  
#join closest player features
data_mod = data_mod %>% left_join(close_features, by = c("game_player_play_id", "frame_id"))



############################################### Catboost built-in CV function ############################################### 


#this is used for identifying whether features should be included in the training or not
#not for tuning 
#the only thing that can be tuned here is iterations
#select the iterations that have the lowest cv error and then select that and tune the remaining pars


#' inputs: feature list or features to exclude - default is to include everything in data_mod (other than responses)
#'         iterations - default 100
#'         player_side - offense or defense models to fit? default is to do both - can set to "offense" or "defense"
#'         response - dir, s, a to fit? default is all - can set to "dir", "s", "a"
#'         prop_cutoff - the proportion of plays complete to fit models on (automatically includes all post throw frames even if before prop_play_complete)
#'         post_throw_only - if TRUE then fits only on post throw frames - default is FALSE
#' outputs 6 saved cv rmse offense and defense for dir, s, a respectively
#' 
#' could do this in parallel
cv_rmse = function(features = "all", exclude_features = FALSE, iterations = 100, side = "both", response = "all", prop_cutoff = 0.625, post_throw_only = FALSE) {
  
  #drop unnecessary features
  unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "num_frames_output", "player_birth_date", 
                            "x", "y", "ball_land_x", "ball_land_y", "player_name", "est_dir", "play_direction", "prev_log_a_diff",
                            "rel_velo_to_ball_land", "rel_acc_to_ball_land", "prev_speed", "prev_acc", "closest_teammate", "closest_opponent")
  data_mod = data_mod %>%
    filter((throw == "post" | throw == "pre" & lead(throw) == "post") | prop_play_complete >= prop_cutoff) #filter by prop_play_complete
  
  #no need to split into training/test - catboost will do that
  cat_df = data_mod %>% 
    filter(est_speed <= 11, #filter out the crazy speeds/accs
           abs(est_acc) <= 15,
           abs(fut_s_diff) <= 1.5,
           abs(fut_a_diff) <= 6) %>%
    filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff)) %>% #remove NA responses
    filter(game_player_play_id != 812) %>% #remove weird play
    select(-any_of(unnnecessary_features))
  
  if(post_throw_only) {#filer to only include frames post throw
    cat_df = cat_df %>% filter(throw == "post")
  }
  if(features != "all") {#filer to only include features in features list
    cat_df = cat_df %>% select(any_of(features))
  }
  if(is.character(exclude_features)) {#exclude identified features
    cat_df = cat_df %>% select(-any_of(exclude_features))
  }
  
  cat_df_o = cat_df %>% filter(player_side == "Offense") %>% select(-c(player_side, starts_with("fut_")))
  cat_df_d = cat_df %>% filter(player_side == "Defense") %>% select(-c(player_side, starts_with("fut_")))
  cat_df_o_labels = cat_df %>% filter(player_side == "Offense") %>% select(fut_dir_diff, fut_s, fut_a_diff)
  cat_df_d_labels = cat_df %>%filter(player_side == "Defense") %>% select(fut_dir_diff, fut_s, fut_a_diff)
  
  #offense pools
  dir_pool_o = catboost.load_pool(cat_df_o, label = cat_df_o_labels$fut_dir_diff)
  s_pool_o = catboost.load_pool(cat_df_o, label = log(cat_df_o_labels$fut_s))
  a_pool_o = catboost.load_pool(cat_df_o, label = cat_df_o_labels$fut_a)
  #defense pools
  dir_pool_d = catboost.load_pool(cat_df_d, label = cat_df_d_labels$fut_dir_diff)
  s_pool_d = catboost.load_pool(cat_df_d, label = log(cat_df_d_labels$fut_s))
  a_pool_d = catboost.load_pool(cat_df_d, label = cat_df_d_labels$fut_a)
  
  #store results
  results_list = list()
  
  #cv models
  if(side %in% c("both", "offense")) {
    if(response %in% c("all", "dir")) {
      dir_o_cv = catboost.cv(dir_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50))
      dir_o_cv = dir_o_cv %>% mutate(response = "dir", side = "offense", iterations = (row_number() - 1)*50)
      results_list = append(results_list, list(dir_o_cv))
    }
    if(response %in% c("all", "s")) {
      speed_o_cv = catboost.cv(s_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                               params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      speed_o_cv = speed_o_cv %>% mutate(response = "speed", side = "offense", iterations = (row_number() - 1)*50)
      results_list = append(results_list, list(speed_o_cv))
    }
    if(response %in% c("all", "a")) {
      acc_o_cv = catboost.cv(a_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      acc_o_cv = acc_o_cv %>% mutate(response = "acc", side = "offense", iterations = (row_number() - 1)*50)
      results_list = append(results_list, list(acc_o_cv))
    }
  }
  
  if(side %in% c("both", "defense")) {
    if(response %in% c("all", "dir")) {
      dir_d_cv = catboost.cv(dir_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      dir_d_cv = dir_d_cv %>% mutate(response = "dir", side = "defense", iterations = (row_number() - 1)*50)
      results_list = append(results_list, list(dir_d_cv))
    }
    if(response %in% c("all", "s")) {
      speed_d_cv = catboost.cv(s_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                               params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      speed_d_cv = speed_d_cv %>% mutate(response = "speed", side = "defense", iterations = (row_number() - 1)*50)
      results_list = append(results_list, list(speed_d_cv))
    }
    if(response %in% c("all", "a")) {
      acc_d_cv = catboost.cv(a_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      acc_d_cv = acc_d_cv %>% mutate(response = "acc", side = "defense", iterations = (row_number() - 1)*50)
      results_list = append(results_list, list(acc_d_cv))
    }
  }
  return(bind_rows(results_list) %>% arrange(response))
}

#' going through each model to remove unnecessary features
#' removing useless features can improve test rmse and the sd down
#' just use post throw only
#' 
#' 
#' 
#' also tune each to the best number of iterations..
#' but is this different since were doing not entirely on post throw?

######### dir_o ######### 
off_close_exclude_features = data_mod %>% select(starts_with("closest_teammate_")) %>% colnames()
#exclude certain features for each model
dir_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position", "log_est_speed", "prev_log_s_diff", "time_elapsed")

# before = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = curr_exclude_features)
# no_closest = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "closest_opponent_dist", "closest_opponent_dir_diff"))
# no_ball_dir_diff = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "ball_land_dir_diff"))
# no_prev_dir_diff = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "prev_dir_diff"))
# no_rel_ball = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "rel_velo_to_ball_land", "rel_acc_to_ball_land"))
# before #4.134
# no_closest #4.133
# no_ball_dir_diff #4.162
# no_prev_dir_diff #5.555
# no_rel_ball #


#tuning iterations
dir_o_it_4000 = cv_rmse(side = "offense", response = "dir", iterations = 4000, exclude_features = c(dir_o_exclude_features, off_close_exclude_features), prop_cutoff = 0.625)

dir_o_it_4000 %>% filter(iterations >= 200) %>% ggplot(aes(x = iterations, y = test.RMSE.mean)) + geom_point()
dir_o_it_4000 %>% filter(test.RMSE.mean == min(test.RMSE.mean))
#900 best



######### dir_d ######### 
#exclude certain features for each model
dir_d_exclude_features = c("player_weight", "player_position", "log_est_speed", "prev_log_s_diff")

# before = cv_rmse(side = "defense", response = "dir", iterations = 100, exclude_features = curr_exclude_features)
# no_closest = cv_rmse(side = "defense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
# before #5.118
# no_closest #5.124

#tuning iterations
dir_d_it_5000 = cv_rmse(side = "defense", response = "dir", iterations = 5000, exclude_features = dir_d_exclude_features, prop_cutoff = 0.625)

dir_d_it_5000 %>% filter(iterations >= 200) %>% ggplot(aes(x = iterations, y = test.RMSE.mean)) + geom_point()
dir_d_it_5000 %>% filter(iterations >= 200) %>% ggplot(aes(x = iterations, y = test.RMSE.std)) + geom_point()
dir_d_it_5000 %>% filter(test.RMSE.mean == min(test.RMSE.mean))

#1950 best



######### speed_o ######### 
speed_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_weight", "player_position", "throw", "est_speed", "prev_speed", "prev_acc")


# before = cv_rmse(side = "offense", response = "s", iterations = 100, exclude_features = curr_exclude_features)
# no_closest = cv_rmse(side = "offense", response = "s", iterations = 100, exclude_features = c(curr_exclude_features, "closest_opponent_dist", "closest_opponent_dir_diff"))
# before #0.1148
# no_closest #0.1148


#tuning iterations
speed_o_it_5000 = cv_rmse(side = "offense", response = "s", iterations = 5000, exclude_features = c(speed_o_exclude_features, off_close_exclude_features))

speed_o_it_5000 %>% filter(iterations >= 1500) %>% ggplot(aes(x = iterations, y = test.RMSE.mean)) + geom_point()
speed_o_it_5000 %>% filter(test.RMSE.mean == min(test.RMSE.mean))

# 4300 best



######### speed_d ######### 
speed_d_exclude_features = c("player_weight", "player_position", "throw", "est_speed", "prev_speed", "prev_acc") #replace log_est_acc with est_acc


# before = cv_rmse(side = "defense", response = "s", iterations = 100, exclude_features = curr_exclude_features)
# no_closest = cv_rmse(side = "defense", response = "s", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
# before #0.1170
# no_closest #0.1172

#tuning iterations
speed_d_it_10000 = cv_rmse(side = "defense", response = "s", iterations = 10000, exclude_features = speed_d_exclude_features)

speed_d_it_10000 %>% filter(iterations >= 3000) %>% ggplot(aes(x = iterations, y = test.RMSE.mean)) + geom_point()
speed_d_it_10000 %>% filter(test.RMSE.mean == min(test.RMSE.mean))

#5500 best



######### acc_o ######### 
acc_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position", "log_est_speed", "prev_log_s_diff")


before = cv_rmse(side = "offense", response = "a", iterations = 100, exclude_features = curr_exclude_features)
no_closest = cv_rmse(side = "offense", response = "a", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
before #1.2559
no_closest #1.2560

#tuning iterations
acc_o_it_9000 = cv_rmse(side = "offense", response = "a", iterations = 9000, exclude_features = c(acc_o_exclude_features, off_close_exclude_features))

acc_o_it_9000 %>% filter(iterations >= 3000) %>% ggplot(aes(x = iterations, y = test.RMSE.mean)) + geom_point()
acc_o_it_9000 %>% filter(test.RMSE.mean == min(test.RMSE.mean))

#5600 best



######### acc_d ######### 
acc_d_exclude_features = c("player_position", "log_est_speed", "prev_log_s_diff")

# before = cv_rmse(side = "defense", response = "a", iterations = 100, exclude_features = FALSE)
# no_closest = cv_rmse(side = "defense", response = "a", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
# before #1.3256
# no_closest #1.3255


#tuning iterations
acc_d_it_12000 = cv_rmse(side = "defense", response = "a", iterations = 12000, exclude_features = acc_d_exclude_features)
 
acc_d_it_12000 %>% filter(iterations >= 6000) %>% ggplot(aes(x = iterations, y = test.RMSE.mean)) + geom_point()
acc_d_it_12000 %>% filter(test.RMSE.mean == min(test.RMSE.mean))

#9850 best


#unfortunately we cant really get feature importance here either - if we want that, need to fit the models individually below - don't need test set


# if were just comparing important features - we dont need to use a lot of iterations, just comapre both set of features on low number of iterations
# I think this is equal playing ground?



#' I think the closest player features matter for the defense a lot, but not so much for the offense
#' maybe only use closest player features for fitting defense models



fit_quick_model = function(response, player_side, exclude_features = FALSE, iterations = 100) {

  unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "num_frames_output", "player_birth_date", 
                            "x", "y", "ball_land_x", "ball_land_y", "player_name", "est_dir", "play_direction")
  
  #no need to split into training/test - catboost will do that
  cat_df = data_mod %>% 
    filter(est_speed <= 11, #filter out the crazy speeds/accs
           abs(est_acc) <= 15,
           abs(fut_s_diff) <= 1.5,
           abs(fut_a_diff) <= 6) %>%
    filter((throw == "post" | throw == "pre" & lead(throw) == "post") | prop_play_complete >= 0.4) %>% #filter prop_play_complete >0.4
    filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff) & !is.na(fut_a)) %>% #remove NA responses
    filter(game_player_play_id != 812) %>% #remove weird play
    select(-all_of(unnnecessary_features))
  
  if(is.character(exclude_features)) {#exclude identified features
    cat_df = cat_df %>% select(-any_of(exclude_features))
  }
  
  if(player_side == "offense") {
    cat_df_labels = cat_df %>% filter(player_side == "Offense") %>% select(fut_dir_diff, fut_s, fut_a_diff, fut_a)
    cat_df = cat_df %>% filter(player_side == "Offense") %>% select(-c(player_side, starts_with("fut_")))
    
    #offense pools
    if(response == "dir") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_dir_diff)
    }
    if(response == "speed") {
      pool = catboost.load_pool(cat_df, label = log(cat_df_labels$fut_s))
    }
    if(response == "acc") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_a)
    }
  }
  
  if(player_side == "defense") {
    cat_df_labels = cat_df %>% filter(player_side == "Defense") %>% select(fut_dir_diff, fut_s, fut_a_diff, fut_a)
    cat_df = cat_df %>% filter(player_side == "Defense") %>% select(-c(player_side, starts_with("fut_")))
    
    #defense pools
    if(response == "dir") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_dir_diff)
    }
    if(response == "speed") {
      pool = catboost.load_pool(cat_df, label = log(cat_df_labels$fut_s))
    }
    if(response == "acc") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_a)
    }
  }
  #fit
  catboost.train(learn_pool = pool, params = list(iterations = iterations, metric_period = 10, 
                                                  od_type = "Iter", od_wait = 50)) 
}

#exclude features
dir_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position")
dir_d_exclude_features = c("player_weight", "player_position")
speed_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_height", "player_weight", "player_position", "throw")
speed_d_exclude_features = c("player_height", "player_weight", "player_position", "est_dir", "throw", "time_elapsed")
acc_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "est_dir", "player_position")
acc_d_exclude_features = c("player_position")

#dir_o
dir_cat_o = fit_quick_model("dir", "offense", exclude_features = dir_o_exclude_features, iterations = 500)
catboost.get_feature_importance(dir_cat_o) #feature importance

#dir_d
dir_cat_d = fit_quick_model("dir", "defense", exclude_features = dir_d_exclude_features)
catboost.get_feature_importance(dir_cat_d) #feature importance

#speed_o
speed_cat_o = fit_quick_model("speed", "offense", exclude_features = speed_o_exclude_features)
catboost.get_feature_importance(speed_cat_o)

#speed_d
speed_cat_d = fit_quick_model("speed", "defense", exclude_features = speed_d_exclude_features)
catboost.get_feature_importance(speed_cat_d) 

#acc_o
acc_cat_o = fit_quick_model("acc", "offense", exclude_features = acc_o_exclude_features)
catboost.get_feature_importance(acc_cat_o) 

#acc_d
acc_cat_d = fit_quick_model("acc", "defense", exclude_features = acc_d_exclude_features)
catboost.get_feature_importance(acc_cat_d) 




