############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(ggrepel)
library(catboost)
library(scattermore)
library(foreach)
library(doFuture)
library(progressr)


############################################### Description ############################################### 
#' predicting x,y from direction, speed, acceleration in current frame and updating (predicting) direction, speed, and acceleration in next frame
#' three models that predict change in dir, s, a each
#' separate dir, s, a models for offense and defense 


############################################### Import data ###############################################

train = read.csv(file = here("data", "train_clean_no_close_player.csv"))
data_mod = read.csv(file = here("data", "data_mod.csv")) %>%
  mutate(across(where(is.character), as.factor)) #for catboost

# #join closest player features
# close_features = read.csv(here("data", "closest_dir_dist_features.csv")) %>%
#   select(game_player_play_id, frame_id, starts_with("closest_"))
# data_mod = data_mod %>% left_join(close_features, by = c("game_player_play_id", "frame_id"))

############################################### Start CV ###############################################

source(here("helper.R"))

set.seed(1999)
num_folds = 5 #80% train, 20% test
game_play_ids = data_mod %>% pull(game_play_id) %>% unique() %>% sort()
num_plays = length(game_play_ids) #14,107 plays

#the cv splits
split = (sample(game_play_ids) %% num_folds) + 1

#the true x,y,dir,s,a values post throw
true_vals = data_mod %>% 
  filter(throw == "post") %>%
  rename(true_dir = est_dir, true_s = est_speed, true_a = est_acc, true_x = x, true_y = y) %>% 
  select(game_player_play_id, frame_id, true_dir, true_s, true_a, true_x, true_y)



############################################### Train Models ###############################################


#data pre processing
data_mod_clean = data_mod %>%  
  filter(est_speed <= 11, #filter out the crazy speeds/accs
         abs(est_acc) <= 15,
         abs(fut_s_diff) <= 1.5,
         abs(fut_a_diff) <= 6) %>%
  #filter(throw == "post" | prop_play_complete >= 0.4) %>% #filter prop_play_complete >0.4
  filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a)) %>% #remove NA responses
  filter(game_player_play_id != 812) #remove weird play
saveRDS(data_mod_clean, file = here("data", "data_mod_clean.RDS"))
#save df and load into each parallel worker so its not transferring huge objects over workers and using tons of memory


#function to train the models on each training fold
#saves models after which can be imported when predicting
#plays are the plays to train on
#cv_split is the 1-5 ids of each play for 5-fold cv
cv_train_models = function(plays, cv_split, test_prop_cutoff = 1, off_train_prop_cutoff = 0.5625, def_train_prop_cutoff = 0.5625) {
  
  #might as well do it in parallel
  registerDoFuture()
  plan(multisession, workers = min(c(num_folds, 10))) #out of 20 cores
  
  #loop through folds
  foreach(fold = 1:num_folds, .packages = c("tidyverse", "doParallel", "catboost")) %dopar% {
    data_mod_clean = readRDS(here("data", "data_mod_clean.RDS")) 
    
    train_plays = plays[cv_split != fold]
    test_plays = plays[cv_split == fold]
    
    #split df
    data_mod_train = data_mod_clean %>% filter(game_play_id %in% train_plays) 
    data_mod_test = data_mod_clean %>% filter(game_play_id %in% test_plays)
    
    
    #fit models on training fold
    
    #drop unnecessary features
    unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "num_frames_output", "player_birth_date", 
                              "x", "y", "ball_land_x", "ball_land_y", "player_name", "est_dir", "play_direction", 
                              "rel_velo_to_ball_land", "rel_acc_to_ball_land", "prev_speed", "prev_acc")
    #exclude certain features for each model
    dir_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")
    dir_d_exclude_features = c("player_weight", "player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")
    speed_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_weight", "player_position", "throw", "est_speed", "prev_speed", "prev_acc")
    speed_d_exclude_features = c("player_weight", "player_position", "throw", "est_speed", "prev_speed", "prev_acc") #replace log_est_acc with est_acc
    acc_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")
    acc_d_exclude_features = c("player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")
    
    cat_train_df = data_mod_train %>% select(-all_of(unnnecessary_features))
    cat_test_df = data_mod_test %>% select(-all_of(unnnecessary_features))
    
    #offense and defense training sets
    train_o = cat_train_df %>% filter(player_side == "Offense", throw == "post" | prop_play_complete >= off_train_prop_cutoff) %>% select(-c(starts_with("fut_"), player_side))
    train_d = cat_train_df %>% filter(player_side == "Defense", throw == "post" | prop_play_complete >= def_train_prop_cutoff) %>% select(-c(starts_with("fut_"), player_side))
    train_o_labels = cat_train_df %>% filter(player_side == "Offense", throw == "post" | prop_play_complete >= off_train_prop_cutoff) %>% select(c(starts_with("fut_")))
    train_d_labels = cat_train_df %>% filter(player_side == "Defense", throw == "post" | prop_play_complete >= def_train_prop_cutoff) %>% select(c(starts_with("fut_")))
    
    #offense and defense test sets
    test_o = cat_test_df %>% filter(player_side == "Offense", throw == "post" | prop_play_complete >= test_prop_cutoff) %>% select(-c(starts_with("fut_"), player_side))
    test_d = cat_test_df %>% filter(player_side == "Offense", throw == "post" | prop_play_complete >= test_prop_cutoff) %>% select(-c(starts_with("fut_"), player_side))
    test_o_labels = cat_test_df %>% filter(player_side == "Offense", throw == "post" | prop_play_complete >= test_prop_cutoff) %>% select(c(starts_with("fut_")))
    test_d_labels = cat_test_df %>% filter(player_side == "Offense", throw == "post" | prop_play_complete >= test_prop_cutoff) %>% select(c(starts_with("fut_")))
     
    #pool train sets
    cat_train_dir_o = catboost.load_pool(train_o %>% select(-any_of(dir_o_exclude_features)), label = train_o_labels$fut_dir_diff)
    cat_train_dir_d = catboost.load_pool(train_d %>% select(-any_of(dir_d_exclude_features)), label = train_d_labels$fut_dir_diff)
    cat_train_speed_o = catboost.load_pool(train_o %>% select(-any_of(speed_o_exclude_features)), label = log(train_o_labels$fut_s)) #try log transform since speed is strictly positive
    cat_train_speed_d = catboost.load_pool(train_d %>% select(-any_of(speed_d_exclude_features)), label = log(train_d_labels$fut_s)) #try log transform since speed is strictly positive
    cat_train_acc_o = catboost.load_pool(train_o %>% select(-any_of(acc_o_exclude_features)), label = train_o_labels$fut_a)
    cat_train_acc_d = catboost.load_pool(train_d %>% select(-any_of(acc_d_exclude_features)), label = train_d_labels$fut_a)
    #pool test sets
    cat_test_dir_o = catboost.load_pool(test_o %>% select(-any_of(dir_o_exclude_features)), label = test_o_labels$fut_dir_diff)
    cat_test_dir_d = catboost.load_pool(test_d %>% select(-any_of(dir_d_exclude_features)), label = test_d_labels$fut_dir_diff)
    cat_test_speed_o = catboost.load_pool(test_o %>% select(-any_of(speed_o_exclude_features)), label = log(test_o_labels$fut_s)) #try log transform since speed is strictly positive
    cat_test_speed_d = catboost.load_pool(test_d %>% select(-any_of(speed_d_exclude_features)), label = log(test_d_labels$fut_s)) #try log transform since speed is strictly positive
    cat_test_acc_o = catboost.load_pool(test_o %>% select(-any_of(acc_o_exclude_features)), label = test_o_labels$fut_a)
    cat_test_acc_d = catboost.load_pool(test_d %>% select(-any_of(acc_d_exclude_features)), label = test_d_labels$fut_a)
  
    #fit
    #just use basic tuning parameters now
    dir_cat_o = catboost.train(learn_pool = cat_train_dir_o, test_pool = cat_test_dir_o, params = list(iterations = 400, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100)) #num iterations to go past min test error before stop
    dir_cat_d = catboost.train(learn_pool = cat_train_dir_d, test_pool = cat_test_dir_d, params = list(iterations = 1300, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    
    speed_cat_o = catboost.train(learn_pool = cat_train_speed_o, test_pool = cat_test_speed_o, params = list(iterations = 2500, logging_level = "Silent",
                                                                                                             od_type = "Iter", od_wait = 100))
    speed_cat_d = catboost.train(learn_pool = cat_train_speed_d, test_pool = cat_test_speed_d, params = list(iterations = 3000, logging_level = "Silent",
                                                                                                             od_type = "Iter", od_wait = 100))
    
    acc_cat_o = catboost.train(learn_pool = cat_train_acc_o, test_pool = cat_test_acc_o, params = list(iterations = 7000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    acc_cat_d = catboost.train(learn_pool = cat_train_acc_d, test_pool = cat_test_acc_d, params = list(iterations = 11000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    #save models
    catboost.save_model(dir_cat_o, model_path = here("models", "experimentation_cv", paste0("fold_", fold), "offense", "dir.cbm"))
    catboost.save_model(dir_cat_d, model_path = here("models", "experimentation_cv", paste0("fold_", fold), "defense", "dir.cbm"))
    catboost.save_model(speed_cat_o, model_path = here("models", "experimentation_cv", paste0("fold_", fold), "offense", "speed.cbm"))
    catboost.save_model(speed_cat_d, model_path = here("models", "experimentation_cv", paste0("fold_", fold), "defense", "speed.cbm"))
    catboost.save_model(acc_cat_o, model_path = here("models", "experimentation_cv",paste0("fold_", fold), "offense", "acc.cbm"))
    catboost.save_model(acc_cat_d, model_path = here("models", "experimentation_cv",paste0("fold_", fold), "defense", "acc.cbm"))
    
    #save features for each model
    if(fold == 1){
      write.table(rownames(dir_cat_o$feature_importances), file = here("models", "experimentation_cv", "cat_features_dir_o.csv"), row.names = FALSE)
      write.table(rownames(dir_cat_d$feature_importances), file = here("models", "experimentation_cv", "cat_features_dir_d.csv"), row.names = FALSE)
      write.table(rownames(speed_cat_o$feature_importances), file = here("models", "experimentation_cv", "cat_features_speed_o.csv"), row.names = FALSE)
      write.table(rownames(speed_cat_d$feature_importances), file = here("models", "experimentation_cv", "cat_features_speed_d.csv"), row.names = FALSE)
      write.table(rownames(acc_cat_o$feature_importances), file = here("models", "experimentation_cv", "cat_features_acc_o.csv"), row.names = FALSE)
      write.table(rownames(acc_cat_d$feature_importances), file = here("models", "experimentation_cv", "cat_features_acc_d.csv"), row.names = FALSE)
    }
  }
}

#get trained models for each cv fold
start = Sys.time()
cv_train_models(game_play_ids, split, test_prop_cutoff = 0.625, off_train_prop_cutoff = 0.6, def_train_prop_cutoff = 0.65)
end = Sys.time()
end-start
plan(sequential) #quit parallel workers



############################################### CV Predict ###############################################

#save all the frames right before throw
data_mod_final_frame = data_mod %>%
  filter(throw == "pre" & lead(throw) == "post")
saveRDS(data_mod_final_frame, file = here("data", "data_mod_final_frame.RDS"))

#function that takes previous models and predicts on each test fold
#plays are the game_play_ids to predict on 
#make sure plays, cv_split is the same for cv_train function above
#pred_subset is the number of rows to predict on in the test folds - make this small to run quicker
#use_best_models is a logical indicating whether to use the current best cv models or not - default is TRUE, use as comparison 
#model_path is which folder for which models you want to use


#' flow of this:
#'  -loop through all (fold, plays)
#'    -loop through all the frames in a num_frames_output
#'    -for each player: predict next x,y and derive new features
#'      -loop through the players in the play post throw
#'      -predict next dir, s, a
cv_predict = function(plays, cv_split, pred_subset = FALSE, model_path = "experimentation_cv", use_closest_features = TRUE, cores = 10, random_pred_subset = FALSE, seed = 1) {
  
  # #set up parallel and progress
  handlers(global = TRUE)
  handlers(handler_txtprogressbar())
  
  registerDoFuture()
  plan(multisession, workers = cores) #out of 20 cores
  
  #features for catboost
  dir_o_features = read.csv(file = here("models", model_path, "cat_features_dir_o.csv")) %>% pull(x)
  dir_d_features = read.csv(file = here("models", model_path, "cat_features_dir_d.csv")) %>% pull(x)
  speed_o_features = read.csv(file = here("models", model_path, "cat_features_speed_o.csv")) %>% pull(x)
  speed_d_features = read.csv(file = here("models", model_path, "cat_features_speed_d.csv")) %>% pull(x)
  acc_o_features = read.csv(file = here("models", model_path, "cat_features_acc_o.csv")) %>% pull(x)
  acc_d_features = read.csv(file = here("models", model_path, "cat_features_acc_d.csv")) %>% pull(x)
  
  all_cat_features = unique(c(dir_o_features, dir_d_features, speed_o_features, speed_d_features, acc_o_features, acc_d_features))
  
  #setting up grid to parallelize over (fold, play)
  set.seed(seed)
  grid = do.call(rbind, lapply(1:num_folds, function(f) {
    test_plays = plays[cv_split == f]
    if(is.numeric(pred_subset)) {
      if(random_pred_subset) {
        test_plays = sample(test_plays, size = pred_subset, replace = FALSE)
      } else {
        test_plays = test_plays[1:pred_subset]
      }
    }
    tibble(fold = f, play = test_plays)
  }))
  
  
  #current play info - includes all players to predict on this play
  data_mod_final_frame = readRDS(here("data", "data_mod_final_frame.RDS")) 
  
  #parallelize here - over the plays
  with_progress({
    p = progressor(steps = nrow(grid)) #progress
    #parallelize over (fold, play)
    results_list = foreach(i = 1:nrow(grid), .packages = c("tidyverse", "catboost")) %dopar% {
      p(sprintf("Iteration %d", i)) #progress 
      
      fold = grid$fold[i]
      play = grid$play[i]
      
      if(!(endsWith(model_path, "cv"))) {#the models for kaggle, no cv folds
        dir_cat_o = catboost.load_model(model_path = here("models", model_path, "offense", "dir.cbm"))
        dir_cat_d = catboost.load_model(model_path = here("models", model_path, "defense", "dir.cbm"))
        speed_cat_o = catboost.load_model(model_path = here("models", model_path, "offense", "speed.cbm"))
        speed_cat_d = catboost.load_model(model_path = here("models", model_path, "defense", "speed.cbm"))
        acc_cat_o = catboost.load_model(model_path = here("models", model_path, "offense", "acc.cbm"))
        acc_cat_d = catboost.load_model(model_path = here("models", model_path, "defense", "acc.cbm"))
      } else {
        #models for this fold - if this is slow its because models are being loaded every single play
        dir_cat_o = catboost.load_model(here("models", model_path, paste0("fold_", fold), "offense", "dir.cbm"))
        dir_cat_d = catboost.load_model(here("models", model_path, paste0("fold_", fold), "defense", "dir.cbm"))
        speed_cat_o = catboost.load_model(here("models", model_path, paste0("fold_", fold), "offense", "speed.cbm"))
        speed_cat_d = catboost.load_model(here("models", model_path, paste0("fold_", fold), "defense", "speed.cbm"))
        acc_cat_o = catboost.load_model(here("models", model_path, paste0("fold_", fold), "offense", "acc.cbm"))
        acc_cat_d = catboost.load_model(here("models", model_path, paste0("fold_", fold), "defense", "acc.cbm"))
      }
    
      curr_play_info = data_mod_final_frame %>% filter(game_play_id == play)
      last_frame_id = curr_play_info$frame_id %>% unique() #last frame id before throw
      num_frames_output = curr_play_info$num_frames_output %>% unique() #number of frames to predict
      player_ids = curr_play_info$game_player_play_id %>% unique() #player ids in the play we need to predict
      
      #loop through frames in play (not in parallel)
      frame_result_list = foreach(output_frame_id = 1:num_frames_output) %do% {
        
        frame = last_frame_id + output_frame_id #current frame
        
        #df to store all results
        curr_frame_all_players = curr_play_info %>%
          #update frame stuff
          mutate(frame_id = frame,
                 prop_play_complete = frame/(last_frame_id + num_frames_output),
                 throw = as.factor(ifelse(output_frame_id > 1, "post", "pre")),
                 time_until_play_complete = ((last_frame_id + num_frames_output) - frame)*0.1,
                 time_elapsed_post_throw = (output_frame_id - 1)*0.1,
                 time_elapsed = frame*0.1)
        
        #if frame is pre throw we already know the features and everything to predict x,y,dir,s,a
        
        #if frame is post throw then we need to update position and dir, s, a based on previous iterations predictions
        if (output_frame_id > 1) { 
          #update current x, y dir, s, a as previous prediction
          prev_frame_all_players = result
          curr_frame_all_players = curr_frame_all_players %>%
            mutate(x = prev_frame_all_players$pred_x,
                   y = prev_frame_all_players$pred_y,
                   est_dir = prev_frame_all_players$pred_dir,
                   est_speed = prev_frame_all_players$pred_s,
                   est_acc = prev_frame_all_players$pred_a,
                   #initialize the preds to be NA
                   pred_x = NA,
                   pred_y = NA,
                   pred_dir = NA,
                   pred_s = NA,
                   pred_a = NA) 
          
          #derive features
          
          #closest player features
          if(use_closest_features) {
            closest_player_features = get_closest_player_min_dist_dir(curr_frame_all_players)
            curr_frame_all_players = curr_frame_all_players %>%
              select(-starts_with("closest_")) %>% #deselect the NA closest player columns so we can merge the right ones
              full_join(closest_player_features, by = c("game_player_play_id"))
          }
          
          #all other features
          #need previous frame to compute lag stuff
          prev_curr_frame_df = curr_frame_all_players %>%
            rbind(prev_frame_all_players) %>%
            arrange(game_player_play_id, frame_id)
          
          #derived other features
          curr_frame_all_players = prev_curr_frame_df %>%
            group_by(game_player_play_id) %>% #make sure to group_by each player
            change_in_kinematics() %>%
            derived_features() %>%
            filter(frame_id == frame) #filter for only current frame
        }
        
        #predict next x,y
        curr_frame_all_players = curr_frame_all_players %>%
          mutate(pred_dist_diff = est_speed*0.1 + est_acc*0.5*0.1^2,
                 pred_x = x + cos(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff,
                 pred_y = y + sin(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff) %>%
          select(-pred_dist_diff)
        
        #predict next dir, s, a
        all_model_features = curr_frame_all_players %>% 
          select(all_of(all_cat_features))#pool all potential features for every model
        full_pool = catboost.load_pool(data = all_model_features) 
        
        #pred dir, s, a
        pred_dir_o = catboost.predict(dir_cat_o, full_pool, thread_count = -1)
        pred_dir_d = catboost.predict(dir_cat_d, full_pool, thread_count = -1)
        pred_s_o = catboost.predict(speed_cat_o, full_pool, thread_count = -1)
        pred_s_d = catboost.predict(speed_cat_d, full_pool, thread_count = -1)
        pred_a_o = catboost.predict(acc_cat_o, full_pool, thread_count = -1)
        pred_a_d = catboost.predict(acc_cat_d, full_pool, thread_count = -1)
        
        curr_frame_all_players = curr_frame_all_players %>%
          mutate(pred_dir = est_dir + ifelse(player_side == "Offense", pred_dir_o[row_number()], pred_dir_d[row_number()]),
                 pred_s = ifelse(player_side == "Offense", pred_s_o[row_number()], pred_s_d[row_number()]),
                 pred_s = exp(pred_s),
                 pred_a = ifelse(player_side == "Offense", pred_a_o[row_number()], pred_a_d[row_number()]))
        
        #same colnames as previous iteration
        if(output_frame_id > 1) {curr_frame_all_players = curr_frame_all_players %>% select(all_of(colnames(result)))}
        
        ### return result ###
        result = curr_frame_all_players 
        result
      }
      bind_rows(frame_result_list)
    }
  })
  bind_rows(results_list)  
}

#train cv
start = Sys.time()
cv_train_models(split, test_prop_cutoff = 0.625, off_train_prop_cutoff = 0.6, def_train_prop_cutoff = 0.65)
end = Sys.time()
end-start
plan(sequential) #quit parallel workers


#predict cv
plan(sequential) #quit parallel workers
start = Sys.time()
results = cv_predict(game_play_ids, split, model_path = "experimentation_cv", pred_subset = 500, use_closest_features = TRUE, cores = 14, random_pred_subset = TRUE, seed = 1) 
end = Sys.time()
end - start
plan(sequential) #quit parallel workers


#random subset of 2500 plays (pred_subset = 500)

#0.625 test prop, 0.60 off prop, 0.65 def prop =====> 0.866 overall, 0.566 offense, 1.02 defense








############################################### RMSE Results ###############################################

### to do:
#' tune prop play complete for each model - can use only 50 pred_subset?
#' tune other parameters for each model like tree depth, etc.
#' figure out why acceleration is still squashed.. (maybe true values are weird?)
#' experiment with other features


#bind results into df
results_comp = results %>%
  left_join(true_vals, by = c("game_player_play_id", "frame_id")) %>% #join true x,y values
  arrange(game_play_id, game_player_play_id, frame_id)

#across entire dataset
results_comp %>% 
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = pred_x, pred_y = pred_y))

### this is experimentation_cv but save it as best_cv for now...


#final_models - 0.762 all 
#kaggle_best - 0.781 all
#exp_final_models - 0.727 all
#final_models - 0.707 all

#the actual rmse is worse than this since were using games we fit the model on in test for final_models
#the cv ones represent the true rmse


#' LOG OF TUNING:
#' 
#' prop_play_complete:
#'    train and test on prop_play_complete > 0.4 - 0.8171
#'    train prop_play_complete > 0.4, test on post throw - 0.822
#'    -you could tune this based on each model, but that seems like so much work
#'    -most potential gain I would guess would be from adding new features/cleaning them up/making them better
#'    -theres probably a lot of potential gain here though
#'    
#' filtering abs(acc)/abs(speed) differences from training/test
#'    -pred_subest = 100, abs(acc<15) - 0.752
#'    -pred_subest = 100, abs(acc<10) - 0.742
#'    -pred_subest = 100, abs(acc<6) - 0.770
#'    -pred_subset = 100, est_speed < 11, abs(est_acc) < 25 - 0.753
#'    -pred_subset = 100, est_speed < 11, abs(est_acc) < 12 - 0.746
#'    -full dataset,      est_speed < 11, abs(est_acc) < 12 - 0.803
#'    
#' 
#' 
#' Overall Best (current)
#'    -0.781 - kaggle_best folder - fit on all features except closest stuff


#by side
results_comp_offense = results_comp %>% filter(player_side == "Offense")
results_comp_defense = results_comp %>% filter(player_side == "Defense")


#offense across entire dataset
results_comp %>% 
  filter(player_side == "Offense") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = pred_x, pred_y = pred_y))
#defense across entire dataset
results_comp %>% 
  filter(player_side == "Defense") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
#defense is much harder to predict 




### RMSE of dir, s, a
sqrt(mean((min_pos_neg_dir(results_comp$pred_dir - results_comp$true_dir))^2)) #18.71 - kaggle_best on pred_subset = 100
sqrt(mean(results_comp$pred_s - results_comp$true_s)^2) #0.032 - kaggle_best
sqrt(mean(results_comp$pred_a - results_comp$true_a)^2) #0.052 - kaggle_best
#sqrt(mean(results_comp$pred_a - results_comp$pred_a + results_comp$pred_a*0.1)^2) #0.052 - kaggle_best


#offense
sqrt(mean((min_pos_neg_dir(results_comp_offense$pred_dir - results_comp_offense$true_dir))^2)) #15.48 - kaggle_best 
#14.57 - final_models all
#15.54 - kaggle_best all


#15.368 - prop 0.1 (0.646 rmse overall)
#14.787 - prop 0.2 (0.643 rmse overall)
#15.422 - prop 0.3 (0.644 rmse overall)
#14.652 - prop 0.4 (0.643 rmse overall)
#15.542 - prop 0.5 (0.644 rmse overall)
#14.771 - prop 0.55 (0.643 rmse overall)
#15.534 - prop 0.6 (0.647 rmse overall)
#15.405 - prop 0.65 (0.643 rmse overall)
#14.313 - prop 0.7 (0.642 rmse overall)
#14.745 - prop 0.8 (0.641 rmse overall)
#14.141 - prop 0.9 (0.644 rmse overall)
#14.053 - prop 1 (0.649 rmse overall) - prop 1 is post throw only


sqrt(mean(results_comp_offense$pred_s - results_comp_offense$true_s)^2) #0.087 - kaggle_best
#0.065 - final_models all
#0.042 - kaggle_best all

sqrt(mean(results_comp_offense$pred_a - results_comp_offense$true_a)^2) #0.202 - kaggle_best
#0.0104 - final_models all
#0.171 - kaggle_best all

#defense
sqrt(mean((min_pos_neg_dir(results_comp_defense$pred_dir - results_comp_defense$true_dir))^2)) #19.88 - kaggle_best
#20.79 - final_models all
#20.74 - kaggle_best all

#19.073 - prop 0.3 (0.661 rmse overall)
#19.435 - prop 0.35 (0.652 rmse overall)
#19.067 - prop 0.38 (0.649 rmse overall)
#19.422 - prop 0.39 (0.653 rmse overall)
#19.041 - prop 0.5 (0.648 rmse overall)
#19.269 - prop 0.55 (0.642 rmse overall)
#19.348 - prop 0.6 (0.641 rmse overall)
#18.963 - prop 0.65 (0.642 rmse overall)
#19.143 - prop 0.7 (0.645 rmes overall)


sqrt(mean(results_comp_defense$pred_s - results_comp_defense$true_s)^2) #0.0091 - kaggle_best
#0.010 - final_models all
#0.011 - kaggle_best all

#0.641 prop 0.4
#0.641 prop 0.5

sqrt(mean(results_comp_defense$pred_a - results_comp_defense$true_a)^2) #0.0102 - kaggle_best 
#0.014 - final_models all
#0.011 - kaggle_best all

#not sure why final_models isnt better on kaggle?


#these are all inflated compared to test rmse when fitting catboost since where using previous predictions as the current dir, s, a and other features
#while in training they use those same derived features but based on true values...



#worst player_plays
results_rmse_player = results_comp %>%
  group_by(game_player_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = pred_x, pred_y = pred_y))
results_rmse_player %>% arrange(desc(rmse))

#worst plays
results_rmse_play = results_comp %>%
  group_by(game_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = pred_x, pred_y = pred_y))
results_rmse_play %>% arrange(desc(rmse))


#plot player rmse
results_rmse_player %>% 
  ggplot(mapping = aes(y = rmse)) +
  geom_boxplot() +
  theme_bw()

#plot play rmse
results_rmse_play %>% 
  ggplot(mapping = aes(y = rmse)) +
  geom_boxplot() +
  theme_bw()



############################################### Visalize Predictions ###############################################


#evaluate dir, s, a results
results_comp %>% select(pred_dir, true_dir) %>% mutate(pred_dir = pred_dir %% 360) %>% summary()
results_comp %>% select(pred_s, true_s) %>% summary()
results_comp %>% select(pred_a, true_a) %>% summary()


#true vs predicted dir (offense)
ggplot(data = results_comp %>% filter(player_side == "Offense"), mapping = aes(x = true_dir, y = pred_dir %% 360)) +
  geom_scattermore(alpha = 0.2) +
  xlim(c(0, 360)) + 
  ylim(c(0, 360)) +
  geom_abline(slope = 1, intercept = 0, colour = "green") +
  theme_bw() +
  labs(x = "True Direction", y = "Predicted Direction")
#true vs predicted dir (defense)
ggplot(data = results_comp %>% filter(player_side == "Defense"), mapping = aes(x = true_dir, y = pred_dir %% 360)) +
  geom_scattermore(alpha = 0.2) +
  xlim(c(0, 360)) + 
  ylim(c(0, 360)) +
  geom_abline(slope = 1, intercept = 0, colour = "green") +
  theme_bw() +
  labs(x = "True Direction", y = "Predicted Direction")


#true vs predicted speed (offense)
ggplot(data = results_comp %>% filter(player_side == "Offense"), mapping = aes(x = true_s, y = pred_s)) +
  geom_scattermore(alpha = 0.2) +
  xlim(c(0, 11)) + 
  ylim(c(0, 11)) +
  theme_bw() +
  geom_abline(slope = 1, intercept = 0, colour = "green") +
  labs(x = "True Speed", y = "Predicted Speed")

#true vs predicted speed (defense)
ggplot(data = results_comp %>% filter(player_side == "Defense"), mapping = aes(x = true_s, y = pred_s)) +
  geom_scattermore(alpha = 0.2) +
  xlim(c(0, 11)) + 
  ylim(c(0, 11)) +
  theme_bw() +
  geom_abline(slope = 1, intercept = 0, colour = "green") +
  labs(x = "True Speed", y = "Predicted Speed")



#true vs predicted acc (offense)
ggplot(data = results_comp %>% filter(player_side == "Offense"), mapping = aes(x = true_a, y = pred_a)) +
  geom_scattermore(alpha = 0.2) +
  xlim(c(-12, 12)) + 
  ylim(c(-12, 12)) +
  geom_abline(slope = 1, intercept = 0, colour = "green") +
  theme_bw() +
  labs(x = "True Acceleration", y = "Predicted Acceleration")

#true vs predicted acc (defense)
ggplot(data = results_comp %>% filter(player_side == "Defense"), mapping = aes(x = true_a, y = pred_a)) +
  geom_scattermore(alpha = 0.2) +
  xlim(c(-12, 12)) + 
  ylim(c(-12, 12)) +
  geom_abline(slope = 1, intercept = 0, colour = "green") +
  theme_bw() +
  labs(x = "True Acceleration", y = "Predicted Acceleration")
#acceleration seems squashed a bit, maybe include higher true vals in training set?

#low accelerations seem to be positively biased?
#high accelerations seem to be negatively biased?
#filter out less of the super high acc?



#pred dir, s, a vs true dir, s, a
group_id = 1
dir_s_a_eval(group_id)
#dir_s_a_eval_player_id(11)

#single player movement
curr_game_player_play_id = results_comp %>% 
  group_by(game_player_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_player_play_id) %>% unique()
#curr_game_player_play_id = 11
plot_player_movement_pred(group_id = curr_game_player_play_id,
                          group_id_preds = results_comp %>% 
                            filter(game_player_play_id == curr_game_player_play_id) %>%
                            select(frame_id, pred_x, pred_y) %>%
                            rename(pred_x = pred_x, pred_y = pred_y))

#multiple players on play
group_id = 1
curr_game_play_id = results_comp %>% 
  group_by(game_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_play_id) %>% unique()
#curr_game_play_id = 812
multi_player_movement_pred(group_id = curr_game_play_id,
                           group_id_preds = results_comp %>%
                             filter(game_play_id == curr_game_play_id) %>%
                             select(game_player_play_id, frame_id, pred_x, pred_y) %>%
                             rename(pred_x = pred_x, pred_y = pred_y))







############################################### Saving final model for Kaggle Submission ###############################################

#quick data cleaning
data_mod_full = data_mod %>% 
  filter(fut_s <= 11, #filter out the crazy speeds/accs
         abs(fut_a) <= 15,
         abs(fut_a_diff) <= 20) %>%
  filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a)) %>% #remove NA responses
  filter(game_player_play_id != 812) #remove weird play

#drop unnecessary features
unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "num_frames_output", "player_birth_date", 
                          "x", "y", "ball_land_x", "ball_land_y", "player_name", "est_dir", "play_direction", 
                          "rel_velo_to_ball_land", "rel_acc_to_ball_land", "prev_speed", "prev_acc")
#exclude certain features for each model
dir_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")
dir_d_exclude_features = c("player_weight", "player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")
speed_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_weight", "player_position", "throw", "est_speed", "prev_speed", "prev_acc")
speed_d_exclude_features = c("player_weight", "player_position", "throw", "est_speed", "prev_speed", "prev_acc") #replace log_est_acc with est_acc
acc_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff", "player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")
acc_d_exclude_features = c("player_position", "log_est_speed", "prev_log_s_diff", "prev_log_a_diff")

### experiment with removing est_acc from speed models - est_acc and log(speed) diff are def correlated

#speed_o, speed_d
#est_acc in old features not in new
#prev_log_a_diff in new not in old


#prop_play_cutoff for each model
dir_o_prop_cutoff = 0.8 #0.8
dir_d_prop_cutoff = 0.65 #0.6
speed_o_prop_cutoff = 0.8 #0.6
speed_d_prop_cutoff = 0.65 #0.6
acc_o_prop_cutoff = 0.8 #0.6
acc_d_prop_cutoff = 0.65 #0.6

#training dfs for each model
dir_o_df = data_mod_full %>% 
  filter(player_side == "Offense") %>%
  filter((throw == "post" | (throw == "pre" & lead(throw) == "post")) | prop_play_complete >= dir_o_prop_cutoff) %>%
  select(-all_of(unnnecessary_features), -starts_with("rel_")) %>%  #### ignore relative velo/acc stuff
  select(-any_of(dir_o_exclude_features))
  
dir_d_df = data_mod_full %>% 
  filter(player_side == "Defense") %>%
  filter((throw == "post" | (throw == "pre" & lead(throw) == "post")) | prop_play_complete >= dir_d_prop_cutoff) %>%
  select(-all_of(unnnecessary_features), -starts_with("rel_")) %>%  #### ignore relative velo/acc stuff
  select(-any_of(dir_d_exclude_features))
  
speed_o_df = data_mod_full %>% 
  filter(player_side == "Offense") %>%
  filter((throw == "post" | (throw == "pre" & lead(throw) == "post")) | prop_play_complete >= speed_o_prop_cutoff) %>%
  select(-all_of(unnnecessary_features), -starts_with("rel_")) %>%  #### ignore relative velo/acc stuff
  select(-any_of(speed_o_exclude_features))
  
speed_d_df = data_mod_full %>% 
  filter(player_side == "Defense") %>%
  filter((throw == "post" | (throw == "pre" & lead(throw) == "post")) | prop_play_complete >= speed_d_prop_cutoff) %>%
  select(-all_of(unnnecessary_features), -starts_with("rel_")) %>%  #### ignore relative velo/acc stuff
  select(-any_of(speed_d_exclude_features))

acc_o_df = data_mod_full %>% 
  filter(player_side == "Offense") %>%
  filter((throw == "post" | (throw == "pre" & lead(throw) == "post")) | prop_play_complete >= acc_o_prop_cutoff) %>%
  select(-all_of(unnnecessary_features), -starts_with("rel_")) %>%  #### ignore relative velo/acc stuff
  select(-any_of(acc_o_exclude_features))

acc_d_df = data_mod_full %>% 
  filter(player_side == "Defense") %>%
  filter((throw == "post" | (throw == "pre" & lead(throw) == "post")) | prop_play_complete >= acc_d_prop_cutoff) %>%
  select(-all_of(unnnecessary_features), -starts_with("rel_")) %>%  #### ignore relative velo/acc stuff
  select(-any_of(acc_d_exclude_features))
 

#pool train sets
cat_train_dir_o = catboost.load_pool(data = dir_o_df %>% select(-c(starts_with("fut_"), player_side)), label = dir_o_df$fut_dir_diff)
cat_train_dir_d = catboost.load_pool(data = dir_d_df %>% select(-c(starts_with("fut_"), player_side)), label = dir_d_df$fut_dir_diff)
cat_train_speed_o = catboost.load_pool(data = speed_o_df %>% select(-c(starts_with("fut_"), player_side)), label = log(speed_o_df$fut_s))
cat_train_speed_d = catboost.load_pool(data = speed_d_df %>% select(-c(starts_with("fut_"), player_side)), label = log(speed_d_df$fut_s))
cat_train_acc_o = catboost.load_pool(data = acc_o_df %>% select(-c(starts_with("fut_"), player_side)), label = acc_o_df$fut_a)
cat_train_acc_d = catboost.load_pool(data = acc_d_df %>% select(-c(starts_with("fut_"), player_side)), label = acc_d_df$fut_a)


#fit final models 

# #for finding how many iterations we need - roughly take avg
# catboost.load_model(model_path = here("models", "best_cv", "fold_1", "defense", "acc.cbm"))
# catboost.load_model(model_path = here("models", "best_cv", "fold_2", "defense", "acc.cbm"))
# catboost.load_model(model_path = here("models", "best_cv", "fold_3", "defense", "acc.cbm"))
# catboost.load_model(model_path = here("models", "best_cv", "fold_4", "defense", "acc.cbm"))
# catboost.load_model(model_path = here("models", "best_cv", "fold_5", "defense", "acc.cbm"))

#fit
dir_cat_o = catboost.train(learn_pool = cat_train_dir_o, params = list(iterations = 400, metric_period = 50)) #400
dir_cat_o$feature_importances
dir_cat_d = catboost.train(learn_pool = cat_train_dir_d, params = list(iterations = 1300, metric_period = 50)) #1300 
dir_cat_d$feature_importances

speed_cat_o = catboost.train(learn_pool = cat_train_speed_o, params = list(iterations = 2500, metric_period = 50)) #2500
speed_cat_o$feature_importances
speed_cat_d = catboost.train(learn_pool = cat_train_speed_d, params = list(iterations = 3000, metric_period = 50)) #3000
speed_cat_d$feature_importances

acc_cat_o = catboost.train(learn_pool = cat_train_acc_o, params = list(iterations = 7000, metric_period = 50)) #7000
acc_cat_o$feature_importances
acc_cat_d = catboost.train(learn_pool = cat_train_acc_d, params = list(iterations = 11000, metric_period = 50)) #11000
acc_cat_d$feature_importances

#save models
catboost.save_model(dir_cat_o, model_path = here("models", "exp_final_models", "offense", "dir.cbm"))
catboost.save_model(dir_cat_d, model_path = here("models", "exp_final_models", "defense", "dir.cbm"))
catboost.save_model(speed_cat_o, model_path = here("models", "exp_final_models", "offense", "speed.cbm"))
catboost.save_model(speed_cat_d, model_path = here("models", "exp_final_models", "defense", "speed.cbm"))
catboost.save_model(acc_cat_o, model_path = here("models", "exp_final_models", "offense", "acc.cbm"))
catboost.save_model(acc_cat_d, model_path = here("models", "exp_final_models", "defense", "acc.cbm"))

#save features
write.table(rownames(dir_cat_o$feature_importances), file = here("models", "exp_final_models", "cat_features_dir_o.csv"), row.names = FALSE)
write.table(rownames(dir_cat_d$feature_importances), file = here("models", "exp_final_models", "cat_features_dir_d.csv"), row.names = FALSE)
write.table(rownames(speed_cat_o$feature_importances), file = here("models", "exp_final_models", "cat_features_speed_o.csv"), row.names = FALSE)
write.table(rownames(speed_cat_d$feature_importances), file = here("models", "exp_final_models", "cat_features_speed_d.csv"), row.names = FALSE)
write.table(rownames(acc_cat_o$feature_importances), file = here("models", "exp_final_models", "cat_features_acc_o.csv"), row.names = FALSE)
write.table(rownames(acc_cat_d$feature_importances), file = here("models", "exp_final_models", "cat_features_acc_d.csv"), row.names = FALSE)



#I think finding the optimal prop cutoff point for each model will be important
#also add more features
#also look at others old catboost posts on kaggle



#check to see whether rel ball land velo/acc is correlated with est_speed/acc 


#maybe make the feature the difference of speed to rel_velo and acc to rel_acc...?








############################################### Comparing preds here to Kaggle ###############################################

test_input = read.csv(here("data", "test_input_with_features.csv"))
test = read.csv(here("data", "test.csv"))

#just do 1 fold cv, ie predict the whole data set
game_play_ids = test_input$game_play_id %>% unique() %>% sort()
test_input_split = rep(1, length(test_input_plays))
num_folds = 1

#just select final frame pre throw
#test_input = test_input %>% group_by(game_player_play_id) %>% slice_tail(n = 1) %>% ungroup()
 
#run CV
#make sure data_mod and cv_split are the same as cv_train()
start = Sys.time()
test_input_results = cv_predict(test_input, test_input_split, model_path = "final_models", pred_subset = 100, use_closest_features = TRUE) 
end = Sys.time()
end - start
plan(sequential) #quit parallel workers



test_input_preds = test_input_results %>% 
  mutate(frame_id = frame_id - max_frame_id) %>%
  select(game_id, nfl_id, play_id, game_play_id, game_player_play_id, frame_id, x, pred_x, y, pred_y)
 

final_preds = test %>% left_join(test_input_preds, by = c("game_id", "nfl_id", "play_id", "frame_id"))
final_preds


#TIME UNTIL PLAY COMPLETE STILL WRONG


