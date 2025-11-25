library(tidyverse)
library(here)
library(catboost)

############################################### Description ############################################### 
#' preliminary file to run before cv training and evaluation
#' to quickly get a sense if new features are beneficial and whether they're worth including
#' basically just runs cv on two different datasets to see if adding features improves cv or not


source(here("helper.R"))
train = read.csv(file = here("data", "train_clean_no_close_player.csv"))
close_features = read.csv(here("data", "closest_dir_dist_features.csv"))
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
cv_rmse = function(features = "all", exclude_features = FALSE, iterations = 100, player_side = "both", response = "all", prop_cutoff = 0.4, post_throw_only = FALSE) {
  
  unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "throw", "num_frames_output",
                            "frame_id", "x", "y", "ball_land_x", "ball_land_y", "player_name")
  data_mod = data_mod %>%
    filter((throw == "post" | throw == "pre" & lead(throw) == "post") | prop_play_complete >= prop_cutoff) #filter by prop_play_complete
  
  if(post_throw_only) {#filer to only include frames post throw
    data_mod = data_mod %>% filter(throw == "post")
  }
  if(features != "all") {#filer to only include features in features list
    data_mod = data_mod %>% select(all_of(features))
  }
  if(is.character(exclude_features)) {#exclude identified features
    data_mod = data_mod %>% select(-all_of(exclude_features))
  }

  #no need to split into training/test - catboost will do that
  cat_df = data_mod %>% 
    filter(abs(fut_s_diff) <= 1, #filter out the crazy speeds/accs for training
           abs(fut_a_diff) <= 5) %>%
    filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff)) %>% #remove NA responses
    select(-all_of(unnnecessary_features))
  
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
  if(player_side %in% c("both", "offense")) {
    if(response %in% c("all", "dir")) {
      dir_o_cv = catboost.cv(dir_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "Iter", od_wait = 25))
      dir_o_cv = data.frame(response = "dir", side = "offense") %>% cbind(dir_o_cv[nrow(dir_o_cv),])
      results_list = append(results_list, list(dir_o_cv))
    }
    if(response %in% c("all", "s")) {
      speed_o_cv = catboost.cv(s_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                               params = list(iterations = iterations, metric_period = 50, od_type = "Iter", od_wait = 25))
      speed_o_cv = data.frame(response = "speed", side = "offense") %>% cbind(speed_o_cv[nrow(speed_o_cv),])
      results_list = append(results_list, list(speed_o_cv))
    }
    if(response %in% c("all", "a")) {
      acc_o_cv = catboost.cv(a_pool_o, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "Iter", od_wait = 25))
      acc_o_cv = data.frame(response = "acc", side = "offense") %>% cbind(acc_o_cv[nrow(acc_o_cv),])
      results_list = append(results_list, list(acc_o_cv))
    }
  }
  
  if(player_side %in% c("both", "defense")) {
    if(response %in% c("all", "dir")) {
      dir_d_cv = catboost.cv(dir_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "Iter", od_wait = 25))
      dir_d_cv = data.frame(response = "dir", side = "defense") %>% cbind(dir_d_cv[nrow(dir_d_cv),])
      results_list = append(results_list, list(dir_d_cv))
    }
    if(response %in% c("all", "s")) {
      speed_d_cv = catboost.cv(s_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                               params = list(iterations = iterations, metric_period = 50, od_type = "Iter", od_wait = 25))
      speed_d_cv = data.frame(response = "speed", side = "defense") %>% cbind(speed_d_cv[nrow(speed_d_cv),])
      results_list = append(results_list, list(speed_d_cv))
    }
    if(response %in% c("all", "a")) {
      acc_d_cv = catboost.cv(a_pool_d, fold_count = 5, type = "Classical", partition_random_seed = 1999,
                             params = list(iterations = iterations, metric_period = 50, od_type = "Iter", od_wait = 25))
      acc_d_cv = data.frame(response = "acc", side = "defense") %>% cbind(acc_d_cv[nrow(acc_d_cv),])
      results_list = append(results_list, list(acc_d_cv))
    }
  }
  return(bind_rows(results_list) %>% arrange(response))
}

#' going through each model to remove unnecessary features
#' removing useless features can improve test rmse and the sd down
#' just use post throw only

######### dir_o ######### 
all_features = cv_rmse(player_side = "offense", response = "dir", iterations = 100,  post_throw_only = TRUE)
no_useless_features = cv_rmse(player_side = "offense", response = "dir", iterations = 100, post_throw_only = TRUE,
                              exclude_features = c("throw", "play_direction", "player_birth_date", "num_frames_output",
                                                   "closest_teammate_dist", "closest_teammate_dir_diff"))
all_features
no_useless_features #since this is the same as all_features we can get rid of them
#' exclude features:
#'  c("throw", "play_direction", "player_birth_date", "num_frames_output", "closest_teammate_dist", "closest_teammate_dir_diff"))


######### dir_d ######### 
all_features = cv_rmse(player_side = "defense", response = "dir", iterations = 100,  post_throw_only = TRUE)
no_useless_features = cv_rmse(player_side = "defense", response = "dir", iterations = 100, post_throw_only = TRUE,
                              exclude_features = c("throw", "player_birth_date", "num_frames_output", "player_position", "play_direction"))
all_features
no_useless_features #since this is the same as all_features we can get rid of them
#' exclude features:
#'  c("throw", "player_birth_date", "num_frames_output", "player_position", "play_direction")


######### speed_o ######### 
all_features = cv_rmse(player_side = "offense", response = "s", iterations = 100,  post_throw_only = TRUE)
no_useless_features = cv_rmse(player_side = "offense", response = "s", iterations = 100, post_throw_only = TRUE,
                              exclude_features = c("throw", "num_frames_output", "play_direction", "player_birth_date", "player_position",
                              "closest_teammate_dist", "closest_teammate_dir_diff"))
all_features
no_useless_features
#' exclude features:
#'  c("throw", "num_frames_output", "play_direction", "player_birth_date", "player_position", "closest_teammate_dist", "closest_teammate_dir_diff"))


######### speed_d ######### 
all_features = cv_rmse(player_side = "defense", response = "s", iterations = 100,  post_throw_only = TRUE)
no_useless_features = cv_rmse(player_side = "defense", response = "s", iterations = 100, post_throw_only = TRUE,
                              exclude_features = c("throw", "num_frames_output", "player_birth_date", "player_position"))
all_features
no_useless_features
#' exclude features:
#'  c("throw", "num_frames_output", "player_birth_date", "player_position"))


######### acc_o ######### 
all_features = cv_rmse(player_side = "offense", response = "a", iterations = 100,  post_throw_only = TRUE)
no_useless_features = cv_rmse(player_side = "offense", response = "a", iterations = 100, post_throw_only = TRUE,
                              exclude_features = c("throw", "num_frames_output", "closest_teammate_dist", "closest_teammate_dir_diff"))
all_features
no_useless_features
#' exclude features:
#'  c("throw", "num_frames_output", "closest_teammate_dist", "closest_teammate_dir_diff")


######### acc_d ######### 
all_features = cv_rmse(player_side = "defense", response = "a", iterations = 100,  post_throw_only = TRUE)
no_useless_features = cv_rmse(player_side = "defense", response = "a", iterations = 100, post_throw_only = TRUE,
                              exclude_features = c("throw", "num_frames_output"))
all_features
no_useless_features
#' exclude features:
#'  c("throw", "num_frames_output")




#I don't think we can tune prop_play_complete here since our validation set is only post throw
#a better cv score here doesn't matter when its automatically evaluating on pre throw frames
#just always set it to a constant and play with the features...

# prop_complete_0 = cv_rmse(player_side = "both", response = "dir", iterations = 100, prop_cutoff = 0)
# prop_complete_0_1 = cv_rmse(player_side = "both", response = "dir", iterations = 100, prop_cutoff = 0.1)
# prop_complete_0_2 = cv_rmse(player_side = "both", response = "dir", iterations = 100, prop_cutoff = 0.2)
# prop_complete_0_3 = cv_rmse(player_side = "both", response = "dir", iterations = 100, prop_cutoff = 0.3)
# prop_complete_0_4 = cv_rmse(player_side = "both", response = "dir", iterations = 100, prop_cutoff = 0.4)
# prop_complete_0_5 = cv_rmse(player_side = "both", response = "dir", iterations = 100, prop_cutoff = 0.5)
# post_throw_only = cv_rmse(player_side = "both", response = "dir", iterations = 100, post_throw_only = TRUE)
# 
# prop_complete_0
# prop_complete_0_1
# prop_complete_0_2
# prop_complete_0_3
# prop_complete_0_4
# prop_complete_0_5
# post_throw_only

#also remember that the best tuning parameters are going to be different for each model
#defense has lower test rmse at 0.4 cutoff but offense is lowest at 0.3 cutoff...


#unfortunately we cant really get feature importance here either - if we want that, need to fit the models individually below - don't need test set


# if were just comparing important features - we dont need to use a lot of iterations, just comapre both set of features on low number of iterations
# I think this is equal playing ground?



#' I think the closest player features matter for the defense a lot, but not so much for the offense
#' maybe only use closest player features for fitting defense models



fit_quick_model = function(response, player_side, exclude_features = FALSE) {

  unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role",
                            "frame_id", "x", "y", "ball_land_x", "ball_land_y", "player_name")
  
  #no need to split into training/test - catboost will do that
  cat_df = data_mod %>% 
    filter(throw == "post") %>%
    filter(abs(fut_s_diff) <= 1, #filter out the crazy speeds/accs for training
           abs(fut_a_diff) <= 6) %>%
    filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff)) %>% #remove NA responses
    select(-all_of(unnnecessary_features))
  
  if(is.character(exclude_features)) {#exclude identified features
    cat_df = cat_df %>% select(-all_of(exclude_features))
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
dir_cat_o = fit_quick_model("dir", "offense", exclude_features = c("throw", "play_direction", "player_birth_date", "num_frames_output", "closest_teammate_dist", "closest_teammate_dir_diff"))
catboost.get_feature_importance(dir_cat_o) %>% format(scientific = FALSE) #feature importance

#dir_d
dir_cat_d = fit_quick_model("dir", "defense", exclude_features = c("throw", "player_birth_date", "num_frames_output", "player_position", "play_direction"))
catboost.get_feature_importance(dir_cat_d) %>% format(scientific = FALSE) #feature importance

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




