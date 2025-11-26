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
cv_rmse = function(features = "all", exclude_features = FALSE, iterations = 100, side = "both", response = "all", prop_cutoff = 0.4, post_throw_only = FALSE) {
  
  unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "num_frames_output", "player_birth_date", 
                            "x", "y", "ball_land_x", "ball_land_y", "player_name", "est_dir", "play_direction")
  data_mod = data_mod %>%
    filter((throw == "post" | throw == "pre" & lead(throw) == "post") | prop_play_complete >= prop_cutoff) #filter by prop_play_complete
  
  if(post_throw_only) {#filer to only include frames post throw
    data_mod = data_mod %>% filter(throw == "post")
  }
  if(features != "all") {#filer to only include features in features list
    data_mod = data_mod %>% select(any_of(features))
  }
  if(is.character(exclude_features)) {#exclude identified features
    data_mod = data_mod %>% select(-any_of(exclude_features))
  }

  #no need to split into training/test - catboost will do that
  cat_df = data_mod %>% 
    filter(est_speed <= 11, #filter out the crazy speeds/accs
           abs(est_acc) <= 15,
           abs(fut_s_diff) <= 1.5,
           abs(fut_a_diff) <= 6) %>%
    filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff)) %>% #remove NA responses
    filter(game_player_play_id != 812) %>% #remove weird play
    select(-any_of(unnnecessary_features))
  
  cat_df_o = cat_df %>% filter(player_side == "Offense") %>% select(-c(player_side, starts_with("fut_")))
  cat_df_d = cat_df %>% filter(player_side == "Defense") %>% select(-c(player_side, starts_with("fut_")))
  cat_df_o_labels = cat_df %>% filter(player_side == "Offense") %>% select(fut_dir_diff, fut_s, fut_a_diff)
  cat_df_d_labels = cat_df %>%filter(player_side == "Defense") %>% select(fut_dir_diff, fut_s, fut_a_diff)
  
  #offense pools
  dir_pool_o = catboost.load_pool(cat_df_o, label = cat_df_o_labels$fut_dir_diff)
  s_pool_o = catboost.load_pool(cat_df_o, label = log(cat_df_o_labels$fut_s))
  a_pool_o = catboost.load_pool(cat_df_o, label = cat_df_o_labels$fut_a_diff)
  #defense pools
  dir_pool_d = catboost.load_pool(cat_df_d, label = cat_df_d_labels$fut_dir_diff)
  s_pool_d = catboost.load_pool(cat_df_d, label = log(cat_df_d_labels$fut_s))
  a_pool_d = catboost.load_pool(cat_df_d, label = cat_df_d_labels$fut_a_diff)
  
  #store results
  results_list = list()
  
  #cv models
  if(side %in% c("both", "offense")) {
    if(response %in% c("all", "dir")) {
      dir_o_cv = catboost.cv(dir_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50))
      dir_o_cv = data.frame(response = "dir", side = "offense") %>% cbind(dir_o_cv[nrow(dir_o_cv),])
      results_list = append(results_list, list(dir_o_cv))
    }
    if(response %in% c("all", "s")) {
      speed_o_cv = catboost.cv(s_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                               params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      speed_o_cv = data.frame(response = "speed", side = "offense") %>% cbind(speed_o_cv[nrow(speed_o_cv),])
      results_list = append(results_list, list(speed_o_cv))
    }
    if(response %in% c("all", "a")) {
      acc_o_cv = catboost.cv(a_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      acc_o_cv = data.frame(response = "acc", side = "offense") %>% cbind(acc_o_cv[nrow(acc_o_cv),])
      results_list = append(results_list, list(acc_o_cv))
    }
  }
  
  if(side %in% c("both", "defense")) {
    if(response %in% c("all", "dir")) {
      dir_d_cv = catboost.cv(dir_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      dir_d_cv = data.frame(response = "dir", side = "defense") %>% cbind(dir_d_cv[nrow(dir_d_cv),])
      results_list = append(results_list, list(dir_d_cv))
    }
    if(response %in% c("all", "s")) {
      speed_d_cv = catboost.cv(s_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                               params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      speed_d_cv = data.frame(response = "speed", side = "defense") %>% cbind(speed_d_cv[nrow(speed_d_cv),])
      results_list = append(results_list, list(speed_d_cv))
    }
    if(response %in% c("all", "a")) {
      acc_d_cv = catboost.cv(a_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "IncToDec", od_wait = 25))
      acc_d_cv = data.frame(response = "acc", side = "defense") %>% cbind(acc_d_cv[nrow(acc_d_cv),])
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
curr_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position") #features currently being excluded in best cv

before = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = curr_exclude_features)
no_closest = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "closest_opponent_dist", "closest_opponent_dir_diff"))
no_ball_dir_diff = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "ball_land_dir_diff"))
no_prev_dir_diff = cv_rmse(side = "offense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "prev_dir_diff"))
before #4.134
no_closest #4.133
no_ball_dir_diff #4.162
no_prev_dir_diff #5.555


#tuning iterations
it_300 = cv_rmse(side = "offense", response = "dir", iterations = 300, exclude_features = curr_exclude_features)
it_350 = cv_rmse(side = "offense", response = "dir", iterations = 350, exclude_features = curr_exclude_features)
it_400 = cv_rmse(side = "offense", response = "dir", iterations = 400, exclude_features = curr_exclude_features)
it_450 = cv_rmse(side = "offense", response = "dir", iterations = 450, exclude_features = curr_exclude_features)
it_500 = cv_rmse(side = "offense", response = "dir", iterations = 500, exclude_features = curr_exclude_features)
it_1000 = cv_rmse(side = "offense", response = "dir", iterations = 1000, exclude_features = curr_exclude_features)

it_300
it_350
it_400
it_450
it_500
it_1000
#400 seems best


######### dir_d ######### 
curr_exclude_features = c("player_weight", "player_position") #features currently being excluded in best cv

before = cv_rmse(side = "defense", response = "dir", iterations = 100, exclude_features = curr_exclude_features)
no_closest = cv_rmse(side = "defense", response = "dir", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
before #5.118
no_closest #5.124

#tuning iterations
it_500 = cv_rmse(side = "defense", response = "dir", iterations = 500, exclude_features = curr_exclude_features)
it_1000 = cv_rmse(side = "defense", response = "dir", iterations = 1000, exclude_features = curr_exclude_features)
it_1500 = cv_rmse(side = "defense", response = "dir", iterations = 1500, exclude_features = curr_exclude_features)

it_500
it_1000
it_1500
#1300 seems best



######### speed_o ######### 
curr_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_height", "player_weight", "player_position", "throw") #features currently being excluded in best cv

before = cv_rmse(side = "offense", response = "s", iterations = 100, exclude_features = curr_exclude_features)
no_closest = cv_rmse(side = "offense", response = "s", iterations = 100, exclude_features = c(curr_exclude_features, "closest_opponent_dist", "closest_opponent_dir_diff"))
before #0.1148
no_closest #0.1148


#tuning iterations
it_500 = cv_rmse(side = "offense", response = "s", iterations = 500, exclude_features = curr_exclude_features)
it_1000 = cv_rmse(side = "offense", response = "s", iterations = 1000, exclude_features = curr_exclude_features)
it_1500 = cv_rmse(side = "offense", response = "s", iterations = 1500, exclude_features = curr_exclude_features)
it_2000 = cv_rmse(side = "offense", response = "s", iterations = 2000, exclude_features = curr_exclude_features)
it_2000 = cv_rmse(side = "offense", response = "s", iterations = 3000, exclude_features = curr_exclude_features)

it_500
it_1000
it_1500
it_2000
it_3000 
# 2500 seems best



######### speed_d ######### 
curr_exclude_features = c("player_height", "player_weight", "player_position", "est_dir", "throw", "time_elapsed") #features currently being excluded in best cv

before = cv_rmse(side = "defense", response = "s", iterations = 100, exclude_features = curr_exclude_features)
no_closest = cv_rmse(side = "defense", response = "s", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
before #0.1170
no_closest #0.1172

#tuning iterations
it_500 = cv_rmse(side = "defense", response = "s", iterations = 500, exclude_features = curr_exclude_features)
it_1000 = cv_rmse(side = "defense", response = "s", iterations = 1000, exclude_features = curr_exclude_features)
it_1500 = cv_rmse(side = "defense", response = "s", iterations = 1500, exclude_features = curr_exclude_features)
it_2000 = cv_rmse(side = "defense", response = "s", iterations = 2000, exclude_features = curr_exclude_features)
it_3000 = cv_rmse(side = "defense", response = "s", iterations = 3000, exclude_features = curr_exclude_features)

it_500
it_1000
it_1500
it_2000
it_3000 
# 3000 seems best


######### acc_o ######### 
curr_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "est_dir") #features currently being excluded in best cv
curr_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "est_dir", "player_position", "player_role") #use just for tuning iterations

before = cv_rmse(side = "offense", response = "a", iterations = 100, exclude_features = curr_exclude_features)
no_closest = cv_rmse(side = "offense", response = "a", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
before #1.2559
no_closest #1.2560

#tuning iterations
it_2000 = cv_rmse(side = "offense", response = "a", iterations = 2000, exclude_features = curr_exclude_features)
it_3000 = cv_rmse(side = "offense", response = "a", iterations = 3000, exclude_features = curr_exclude_features)
it_5000 = cv_rmse(side = "offense", response = "a", iterations = 5000, exclude_features = curr_exclude_features)
it_7500 = cv_rmse(side = "offense", response = "a", iterations = 7500, exclude_features = curr_exclude_features)

it_2000
it_3000
it_5000
it_7500

#7000 seems best



######### acc_d ######### 
curr_exclude_features = c("player_position") #features currently being excluded in best cv

before = cv_rmse(side = "defense", response = "a", iterations = 100, exclude_features = FALSE)
no_closest = cv_rmse(side = "defense", response = "a", iterations = 100, exclude_features = c(curr_exclude_features, "closest_teammate_dist", "closest_teammate_dir_diff", "closest_opponent_dist", "closest_opponent_dir_diff"))
before #1.3256
no_closest #1.3255


#tuning iterations
it_2000 = cv_rmse(side = "defense", response = "a", iterations = 2000, exclude_features = curr_exclude_features)
it_3000 = cv_rmse(side = "defense", response = "a", iterations = 3000, exclude_features = curr_exclude_features)
it_5000 = cv_rmse(side = "defense", response = "a", iterations = 5000, exclude_features = curr_exclude_features)
it_7500 = cv_rmse(side = "defense", response = "a", iterations = 7500, exclude_features = curr_exclude_features)

it_2000
it_3000
it_5000
it_7500

#like 10000 best


#unfortunately we cant really get feature importance here either - if we want that, need to fit the models individually below - don't need test set


# if were just comparing important features - we dont need to use a lot of iterations, just comapre both set of features on low number of iterations
# I think this is equal playing ground?



#' I think the closest player features matter for the defense a lot, but not so much for the offense
#' maybe only use closest player features for fitting defense models



fit_quick_model = function(response, player_side, exclude_features = FALSE) {

  unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "num_frames_output", "player_birth_date", 
                            "x", "y", "ball_land_x", "ball_land_y", "player_name", "est_dir", "play_direction")
  
  #no need to split into training/test - catboost will do that
  cat_df = data_mod %>% 
    filter(est_speed <= 11, #filter out the crazy speeds/accs
           abs(est_acc) <= 15,
           abs(fut_s_diff) <= 1.5,
           abs(fut_a_diff) <= 6) %>%
    filter((throw == "post" | throw == "pre" & lead(throw) == "post") | prop_play_complete >= 0.4) %>% #filter prop_play_complete >0.4
    filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff)) %>% #remove NA responses
    filter(game_player_play_id != 812) %>% #remove weird play
    select(-all_of(unnnecessary_features))
  
  if(is.character(exclude_features)) {#exclude identified features
    cat_df = cat_df %>% select(-any_of(exclude_features))
  }
  
  if(player_side == "offense") {
    cat_df_labels = cat_df %>% filter(player_side == "Offense") %>% select(fut_dir_diff, fut_s, fut_a_diff)
    cat_df = cat_df %>% filter(player_side == "Offense") %>% select(-c(player_side, starts_with("fut_")))
    
    #offense pools
    if(response == "dir") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_dir_diff)
    }
    if(response == "speed") {
      pool = catboost.load_pool(cat_df, label = log(cat_df_labels$fut_s))
    }
    if(response == "acc") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_a_diff)
    }
  }
  
  if(player_side == "defense") {
    cat_df_labels = cat_df %>% filter(player_side == "Defense") %>% select(fut_dir_diff, fut_s, fut_a_diff)
    cat_df = cat_df %>% filter(player_side == "Defense") %>% select(-c(player_side, starts_with("fut_")))
    
    #defense pools
    if(response == "dir") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_dir_diff)
    }
    if(response == "speed") {
      pool = catboost.load_pool(cat_df, label = log(cat_df_labels$fut_s))
    }
    if(response == "acc") {
      pool = catboost.load_pool(cat_df, label = cat_df_labels$fut_a_diff)
    }
  }
  #fit
  catboost.train(learn_pool = pool, params = list(iterations = 100, metric_period = 10, 
                                                  od_type = "Iter", od_wait = 50)) 
}

#dir_o
dir_cat_o = fit_quick_model("dir", "offense", exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff"))
catboost.get_feature_importance(dir_cat_o) #feature importance

#dir_d
dir_cat_d = fit_quick_model("dir", "defense", exclude_features = c("player_position"))
catboost.get_feature_importance(dir_cat_d) #feature importance

#speed_o
speed_cat_o = fit_quick_model("speed", "offense", exclude_features = c("throw", "num_frames_output", "play_direction", "player_birth_date", "closest_teammate_dist", "closest_teammate_dir_diff", "player_height", "player_position"))
catboost.get_feature_importance(speed_cat_o) %>% format(scientific = FALSE) #feature importance

#speed_d
speed_cat_d = fit_quick_model("speed", "defense", exclude_features = c("throw", "num_frames_output", "player_birth_date", "player_position"))
catboost.get_feature_importance(speed_cat_d) %>% format(scientific = FALSE) #feature importance

#acc_o
acc_cat_o = fit_quick_model("acc", "offense", exclude_features = c("throw", "num_frames_output", "closest_teammate_dist", "closest_teammate_dir_diff"))
catboost.get_feature_importance(acc_cat_o) %>% format(scientific = FALSE) #feature importance

#acc_d
acc_cat_d = fit_quick_model("acc", "defense", exclude_features = c("throw", "num_frames_output"))
catboost.get_feature_importance(acc_cat_d) %>% format(scientific = FALSE) #feature importance




