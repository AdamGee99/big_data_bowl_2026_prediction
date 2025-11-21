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
data_mod = read.csv(file = here("data", "data_mod_no_close_player.csv")) %>%
  mutate(across(where(is.character), as.factor)) #for catboost

#join closest player features
close_features = read.csv(here("data", "closest_dir_dist_features.csv"))
data_mod = data_mod %>% left_join(close_features, by = c("game_player_play_id", "frame_id"))

############################################### Start CV ###############################################

source(here("helper.R"))

set.seed(1999)
num_folds = 5 #80% train, 20% test
game_play_ids = data_mod %>% pull(game_play_id) %>% unique() %>% sort()
num_plays = length(game_play_ids) #14,107 plays

#the cv splits
split = (sample(game_play_ids) %% num_folds) + 1



############################################### Train Models ###############################################


#function to train the models on each training fold
#saves models after which can be imported when predicting
#df is data_mod
cv_train_models = function(df, cv_split) {
  
  #might as well do it in parallel
  registerDoFuture()
  plan(multisession, workers = min(c(num_folds, 10))) #out of 20 cores
  
  #loop through folds
  foreach(fold = 1:num_folds, .packages = c("tidyverse", "doParallel", "catboost")) %dopar% {
    train_plays = game_play_ids[split != fold]
    test_plays = game_play_ids[split == fold]
    
    #data pre processing
    df = df %>%  
      filter(abs(fut_s_diff) <= 1, #filter out the crazy speeds/accs for training
             abs(fut_a_diff) <= 6) %>%
      filter((throw == "post" | throw == "pre" & lead(throw) == "post") | prop_play_complete >= 0.4) %>% #filter prop_play_complete >0.4
      filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff)) #remove NA responses
    
    #split df
    data_mod_train = df %>% filter(game_play_id %in% train_plays) 
    data_mod_test = df %>% filter(game_play_id %in% test_plays)
    
    
    #fit models on training fold
    
    #drop unnecessary features
    unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role", "throw", "num_frames_output", "play_direction", "player_birth_date", 
                              "frame_id", "x", "y", "ball_land_x", "ball_land_y", "player_name")
    cat_train_df = data_mod_train %>% select(-all_of(unnnecessary_features))
    cat_test_df = data_mod_test %>%select(-all_of(unnnecessary_features))
    
    #offense and defense training sets
    train_o = cat_train_df %>% filter(player_side == "Offense") %>% select(-c(starts_with("fut_"), player_side))
    train_d = cat_train_df %>% filter(player_side == "Defense") %>% select(-c(starts_with("fut_"), player_side))
    train_o_labels = cat_train_df %>% filter(player_side == "Offense") %>% select(c(starts_with("fut_")))
    train_d_labels = cat_train_df %>% filter(player_side == "Defense") %>% select(c(starts_with("fut_")))
    
    #test sets
    #offense and defense training sets
    test_o = cat_test_df %>% filter(player_side == "Offense") %>% select(-c(starts_with("fut_"), player_side))
    test_d = cat_test_df %>% filter(player_side == "Defense") %>% select(-c(starts_with("fut_"), player_side))
    test_o_labels = cat_test_df %>% filter(player_side == "Offense") %>% select(c(starts_with("fut_")))
    test_d_labels = cat_test_df %>% filter(player_side == "Defense") %>% select(c(starts_with("fut_")))
    
    #exclude certain features for each model
    dir_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff")
    dir_d_exclude_features = c("player_position")
    speed_o_exclude_features = c("player_position", "closest_teammate_dist", "closest_teammate_dir_diff")
    speed_d_exclude_features = c("player_position")
    acc_o_exclude_features = c("closest_teammate_dist", "closest_teammate_dir_diff")
    #acc_d_exclude_features = c()  #none for acc_d
    
    #pool train sets
    cat_train_dir_o = catboost.load_pool(train_o %>% select(-any_of(dir_o_exclude_features)), label = train_o_labels$fut_dir_diff)
    cat_train_dir_d = catboost.load_pool(train_d %>% select(-any_of(dir_d_exclude_features)), label = train_d_labels$fut_dir_diff)
    cat_train_speed_o = catboost.load_pool(train_o %>% select(-any_of(speed_o_exclude_features)), label = log(train_o_labels$fut_s)) #try log transform since speed is strictly positive
    cat_train_speed_d = catboost.load_pool(train_d %>% select(-any_of(speed_d_exclude_features)), label = log(train_d_labels$fut_s)) #try log transform since speed is strictly positive
    cat_train_acc_o = catboost.load_pool(train_o %>% select(-any_of(acc_o_exclude_features)), label = train_o_labels$fut_a_diff)
    cat_train_acc_d = catboost.load_pool(train_d, label = train_d_labels$fut_a_diff)
    #pool test sets
    cat_test_dir_o = catboost.load_pool(test_o %>% select(-any_of(dir_o_exclude_features)), label = test_o_labels$fut_dir_diff)
    cat_test_dir_d = catboost.load_pool(test_d %>% select(-any_of(dir_d_exclude_features)), label = test_d_labels$fut_dir_diff)
    cat_test_speed_o = catboost.load_pool(test_o %>% select(-any_of(speed_o_exclude_features)), label = log(test_o_labels$fut_s)) #try log transform since speed is strictly positive
    cat_test_speed_d = catboost.load_pool(test_d %>% select(-any_of(speed_d_exclude_features)), label = log(test_d_labels$fut_s)) #try log transform since speed is strictly positive
    cat_test_acc_o = catboost.load_pool(test_o %>% select(-any_of(acc_o_exclude_features)), label = test_o_labels$fut_a_diff)
    cat_test_acc_d = catboost.load_pool(test_d, label = test_d_labels$fut_a_diff)
    
    #save features for each model
    if(fold == 1){
      write.table(colnames(train_o %>% select(-any_of(dir_o_exclude_features))), file  = here("models", "cat_features_dir_o.csv"), row.names = FALSE)
      write.table(colnames(train_o %>% select(-any_of(dir_d_exclude_features))), file  = here("models", "cat_features_dir_d.csv"), row.names = FALSE)
      write.table(colnames(train_o %>% select(-any_of(speed_o_exclude_features))), file  = here("models", "cat_features_speed_o.csv"), row.names = FALSE)
      write.table(colnames(train_o %>% select(-any_of(speed_d_exclude_features))), file  = here("models", "cat_features_speed_d.csv"), row.names = FALSE)
      write.table(colnames(train_o %>% select(-any_of(acc_o_exclude_features))), file  = here("models", "cat_features_acc_o.csv"), row.names = FALSE)
      write.table(colnames(train_o), file  = here("models", "cat_features_acc_d.csv"), row.names = FALSE)
    }
    
    #fit
    #just use basic tuning parameters now
    dir_cat_o = catboost.train(learn_pool = cat_train_dir_o, test_pool = cat_test_dir_o, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100)) #num iterations to go past min test error before stop
    dir_cat_d = catboost.train(learn_pool = cat_train_dir_d, test_pool = cat_test_dir_d, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    
    speed_cat_o = catboost.train(learn_pool = cat_train_speed_o, test_pool = cat_test_speed_o, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                             od_type = "Iter", od_wait = 100))
    speed_cat_d = catboost.train(learn_pool = cat_train_speed_d, test_pool = cat_test_speed_d, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                             od_type = "Iter", od_wait = 100))
    
    acc_cat_o = catboost.train(learn_pool = cat_train_acc_o, test_pool = cat_test_acc_o, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    acc_cat_d = catboost.train(learn_pool = cat_train_acc_d, test_pool = cat_test_acc_d, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    #save models
    catboost.save_model(dir_cat_o, model_path = here("models", paste0("fold_", fold), "offense", "dir.cbm"))
    catboost.save_model(dir_cat_d, model_path = here("models", paste0("fold_", fold), "defense", "dir.cbm"))
    catboost.save_model(speed_cat_o, model_path = here("models", paste0("fold_", fold), "offense", "speed.cbm"))
    catboost.save_model(speed_cat_d, model_path = here("models", paste0("fold_", fold), "defense", "speed.cbm"))
    catboost.save_model(acc_cat_o, model_path = here("models", paste0("fold_", fold), "offense", "acc.cbm"))
    catboost.save_model(acc_cat_d, model_path = here("models", paste0("fold_", fold), "defense", "acc.cbm"))
  }
}

#get trained models for each cv fold
start = Sys.time()
cv_train_models(data_mod, split)
end = Sys.time()
end-start
plan(sequential) #quit parallel workers

#takes 15 min for 1000 iterations each model


### what if we filter throw == "post" only for test set...


############################################### CV Predict ###############################################


#features for catboost
dir_o_features = read.csv(file = here("models", "cat_features_dir_o.csv")) %>% pull(x)
dir_d_features = read.csv(file = here("models", "cat_features_dir_d.csv")) %>% pull(x)
speed_o_features = read.csv(file = here("models", "cat_features_speed_o.csv")) %>% pull(x)
speed_d_features = read.csv(file = here("models", "cat_features_speed_d.csv")) %>% pull(x)
acc_o_features = read.csv(file = here("models", "cat_features_acc_o.csv")) %>% pull(x)
acc_d_features = read.csv(file = here("models", "cat_features_acc_d.csv")) %>% pull(x)

#function that takes previous models and predicts on each test fold
#df is data_mod
#pred_subset is the number of rows to predict on in the test folds - make this small to run quicker
cv_predict = function(df, pred_subset = FALSE) {
  
  #set up parallel and progress
  handlers(global = TRUE)
  handlers("progress")
  
  registerDoFuture()
  plan(multisession, workers = 15) #out of 20 cores
  
  
  #loop through folds
  foreach(fold = 1:num_folds, .combine = rbind, .packages = c("tidyverse", "doParallel", "catboost")) %do% {
    print(paste0("Starting fold ", fold, ". ", 100*((fold - 1)/num_folds), "% done")) #see progress

    #plays we need to predict on
    test_plays = game_play_ids[split == fold]
    data_mod_test = df %>% 
      filter(game_play_id %in% test_plays,
             throw == "pre" & lead(throw) == "post") #select the final frame before throw
    
    #predict on subset of test folds
    if(is.numeric(pred_subset)) {
       test_plays = test_plays[1:pred_subset]
    }
    
    #load models for this fold
    dir_cat_o = catboost.load_model(model_path = here("models", paste0("fold_", fold), "offense", "dir.cbm"))
    dir_cat_d = catboost.load_model(model_path = here("models", paste0("fold_", fold), "defense", "dir.cbm"))
    speed_cat_o = catboost.load_model(model_path = here("models", paste0("fold_", fold), "offense", "speed.cbm"))
    speed_cat_d = catboost.load_model(model_path = here("models", paste0("fold_", fold), "defense", "speed.cbm"))
    acc_cat_o = catboost.load_model(model_path = here("models", paste0("fold_", fold), "offense", "acc.cbm"))
    acc_cat_d = catboost.load_model(model_path = here("models", paste0("fold_", fold), "defense", "acc.cbm"))
    
    #test using final models - the same oens used for kaggle submission
    # dir_cat_o = catboost.load_model(model_path = here("models", "final_models", "offense", "dir.cbm"))
    # dir_cat_d = catboost.load_model(model_path = here("models", "final_models", "defense", "dir.cbm"))
    # speed_cat_o = catboost.load_model(model_path = here("models", "final_models", "offense", "speed.cbm"))
    # speed_cat_d = catboost.load_model(model_path = here("models", "final_models", "defense", "speed.cbm"))
    # acc_cat_o = catboost.load_model(model_path = here("models", "final_models", "offense", "acc.cbm"))
    # acc_cat_d = catboost.load_model(model_path = here("models", "final_models", "defense", "acc.cbm"))
    
    #' flow of this:
    #'  -loop through all the plays
    #'    -loop through all the frames in a num_frames_output
    #'    -for each player: predict next x,y and derive new features
    #'      -loop through the players in the play post throw
    #'      -predict next dir, s, a
  
    #parallelize here - over the plays
    with_progress({
      p = progressor(steps = length(test_plays)) #progress
      
      results = foreach(play = test_plays, .combine = rbind, .packages = c("tidyverse", "doParallel", "catboost")) %dopar% {
        p(sprintf("Iteration %d", play)) #progress 
        
        #current play info - includes all players to predict on this play
        curr_play_info = data_mod_test %>% filter(game_play_id == play)
        last_frame_id = curr_play_info$frame_id %>% unique() #last frame id before throw
        num_frames_output = curr_play_info$num_frames_output %>% unique() #number of frames to predict
        player_ids = curr_play_info$game_player_play_id %>% unique() #player ids in the play we need to predict
        
        #loop through frames in play (not in parallel)
        foreach(output_frame_id = 1:num_frames_output, .combine = rbind) %do% {
          
          frame = last_frame_id + output_frame_id #current frame
          
          #df to store all results
          curr_frame_all_players = curr_play_info %>%
            #update frame stuff
            mutate(frame_id = frame,
                   prop_play_complete = frame/(last_frame_id + num_frames_output),
                   throw = as.factor(ifelse(output_frame_id > 1, "post", "pre")))
          
          #if frame is pre throw we already know the features and everything to predict x,y,dir,s,a
          
          #if frame is post throw then we need to update position and dir, s, a based on previous iterations predictions
          if (unique(curr_frame_all_players$throw) == "post") { 
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
            closest_player_features = get_closest_player_min_dist_dir(curr_frame_all_players) 
            curr_frame_all_players = curr_frame_all_players %>% 
              select(-starts_with("closest_")) %>% #deselect the NA closest player columns so we can merge the right ones
              full_join(closest_player_features, by = c("game_player_play_id"))
            
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
          curr_frame_all_players = foreach(player = player_ids, .combine = rbind) %do% { #need to do this in a loop since catboost load pool thing
            
            #return row with predicted dir, s, a
            curr_frame_player = curr_frame_all_players %>% 
              filter(game_player_play_id == player) #predict on single player
              
            #load in models
            if(unique(curr_frame_player$player_side) == "Offense") {
              dir_cat = dir_cat_o
              speed_cat = speed_cat_o
              acc_cat = acc_cat_o
              
              #catboost pool
              dir_pool = curr_frame_player %>% select(all_of(dir_o_features)) %>% catboost.load_pool()
              speed_pool = curr_frame_player %>% select(all_of(speed_o_features)) %>% catboost.load_pool()
              acc_pool = curr_frame_player %>% select(all_of(acc_o_features)) %>% catboost.load_pool()
            } else {
              dir_cat = dir_cat_d
              speed_cat = speed_cat_d
              acc_cat = acc_cat_d
              
              #catboost pool
              dir_pool = curr_frame_player %>% select(all_of(dir_d_features)) %>% catboost.load_pool()
              speed_pool = curr_frame_player %>% select(all_of(speed_d_features)) %>% catboost.load_pool()
              acc_pool = curr_frame_player %>% select(all_of(acc_d_features)) %>% catboost.load_pool()
            }
            #predict
            curr_frame_player = curr_frame_player %>%  
              mutate(pred_dir = est_dir + catboost.predict(dir_cat, dir_pool),
                     pred_s = catboost.predict(speed_cat, speed_pool), 
                     pred_s = exp(pred_s), #exponentiate back to original scale
                     pred_a = est_acc + catboost.predict(acc_cat, acc_pool))
            
            #same colnames as previous iteration
            if(curr_frame_player$throw != "pre") {curr_frame_all_players = curr_frame_all_players %>% select(all_of(colnames(result)))} 
            
            curr_frame_player
          }
          
          ### return result ###
          result = curr_frame_all_players
          result
        }
      }
    })
  }
}

#run CV
start = Sys.time()
results = cv_predict(data_mod) 
end = Sys.time()
end - start
plan(sequential) #quit parallel workers

#entire dataset takes 29 min


### experiment with this - figure out the best method - compare rmse at the end
#' 1. predict x,y first, then use future x,y to predict future dir, s, a
#' 2. predict dir,s,a first, then use future dir,s,a to predict future x,y
#' 3. do both and take the average prediciton - do this iteratively, take the mean at every step...

### another thing that would probably improve predictions:
#' generate first predictions using only lag x,y 
#' after you get the predictions, use those to get better estimates of dir, s, a on the current frame
#' by calculating dir,s,a over prev_frame -> leading frame, this gives estimate of kinematics at the actual current frame
#' 
#' can keep doing this until convergence... 
#' but how computationally feasible is this
#' 
#' 


#' TO DO 
#' 
#' python def has a library that can call R code so that's how the API will handle it
#' can save catboost models as a .parquet file so python can run it quickly




############################################### Results ###############################################


#the true x,y,dir,s,a values
true_vals = data_mod %>% 
  filter(throw == "post") %>%
  rename(true_dir = est_dir, true_s = est_speed, true_a = est_acc, true_x = x, true_y = y) %>% 
  select(game_player_play_id, frame_id, true_dir, true_s, true_a, true_x, true_y)

#bind results into df
results_comp = results %>%
  left_join(true_vals, by = c("game_player_play_id", "frame_id")) %>% #join true x,y values
  arrange(game_play_id, game_player_play_id, frame_id)


############################################### RMSE ###############################################

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

#across entire dataset
results_comp %>% 
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = pred_x, pred_y = pred_y))

#0.783 best

#fit on all -                     - 1.111


#fit on prop_play_complete > 0.4  - 1.083

#fit on prop_play_complete > 0.4  - 1.035
#    and separate off,def models 


#fit on throw == "post"           - 1.154

#same as above but filtering out weird data - 0.730
#same as above but band-aid fix to negative speeds - 0.729


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
sqrt(mean((min_pos_neg_dir(results_comp$pred_dir - results_comp$true_dir))^2)) #20.37 rmse for direction
sqrt(mean(results_comp$pred_s - results_comp$true_s)^2) #0.069 rmse for speed
sqrt(mean(results_comp$pred_a - results_comp$true_a)^2) #0.011 rmse for acceleration

#these are all inflated compared to test rmse when fitting catboost since where using previous predictions as the current dir, s, a and other features
#while in training they use those same derived features but based on true values...






############################################### Visalize Predictions ###############################################


#evaluate dir, s, a results
results_comp %>% select(pred_dir, true_dir) %>% summary()
results_comp %>% select(pred_dir, true_dir) %>% mutate(pred_dir = pred_dir %% 360) %>% summary()
results_comp %>% select(pred_s, true_s) %>% summary()
results_comp %>% select(pred_a, true_a) %>% summary()


#true vs predicted dir
ggplot(data = results_comp, mapping = aes(x = true_dir, y = pred_dir)) +
  geom_scattermore(alpha = 0.2) +
  xlim(c(0, 360)) + 
  #ylim(c(-20,20)) +
  theme_bw() +
  labs(x = "True Direction", y = "Predicted Direction")

#true vs predicted speed
ggplot(data = results_comp, mapping = aes(x = true_s, y = pred_s)) +
  geom_scattermore(alpha = 0.1) +
  xlim(c(min(c(results_comp$pred_s, results_comp$true_s))), 
       max(c(results_comp$pred_s, results_comp$true_s))) + 
  ylim(c(min(c(results_comp$pred_s, results_comp$true_s))), 
       max(c(results_comp$pred_s, results_comp$true_s))) +
  theme_bw() +
  labs(x = "True Speed", y = "Predicted Speed")

#true vs predicted acc
ggplot(data = results_comp, mapping = aes(x = true_a, y = pred_a)) +
  geom_scattermore(alpha = 0.1) +
  xlim(c(min(c(results_comp$pred_a, results_comp$true_a))), 
       max(c(results_comp$pred_a, results_comp$true_a))) + 
  ylim(c(min(c(results_comp$pred_a, results_comp$true_a))), 
       max(c(results_comp$pred_a, results_comp$true_a))) +
  theme_bw() + xlim(c(-10, 10)) + ylim(c(-10,10)) +
  labs(x = "True Acceleration", y = "Predicted Acceleration")
#acceleration seems squashed a bit, maybe include higher true vals in training set?



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
group_id = 2
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

#split df
data_mod_full = data_mod %>% 
  filter(abs(fut_s_diff) <= 1, #filter out the crazy speeds/accs for training
         abs(fut_a_diff) <= 6) %>%
  filter(!is.na(fut_dir_diff) & !is.na(fut_s) & !is.na(fut_a_diff), #remove NA responses
         game_player_play_id != 812) #remove weird play

#drop unnecessary features


unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role",
                          "frame_id", "x", "y", "ball_land_x", "ball_land_y", "player_name")
############# ignore closest player features for now ###############
cat_train_df = data_mod_full %>% select(-all_of(unnnecessary_features), -starts_with("closest_"))

#offense and defense training sets
train_o = cat_train_df %>% filter(player_side == "Offense") %>% select(-c(starts_with("fut_"), player_side))
train_d = cat_train_df %>% filter(player_side == "Defense") %>% select(-c(starts_with("fut_"), player_side))
train_o_labels = cat_train_df %>% filter(player_side == "Offense") %>% select(c(starts_with("fut_")))
train_d_labels = cat_train_df %>% filter(player_side == "Defense") %>% select(c(starts_with("fut_")))


#df in right type for catboost
#train sets
cat_train_dir_o = catboost.load_pool(train_o, label = train_o_labels$fut_dir_diff)
cat_train_dir_d = catboost.load_pool(train_d, label = train_d_labels$fut_dir_diff)
cat_train_speed_o = catboost.load_pool(train_o, label = log(train_o_labels$fut_s)) #try log transform since speed is strictly positive
cat_train_speed_d = catboost.load_pool(train_d, label = log(train_d_labels$fut_s)) #try log transform since speed is strictly positive
cat_train_acc_o = catboost.load_pool(train_o, label = train_o_labels$fut_a_diff)
cat_train_acc_d = catboost.load_pool(train_d, label = train_d_labels$fut_a_diff)


#fit final models 

#use the same parameters as above - found from cv
#right now just copying the avg iterations
#figure out how to save more tuning paramter values

dir_cat_o = catboost.train(learn_pool = cat_train_dir_o, params = list(iterations = 100))
dir_cat_d = catboost.train(learn_pool = cat_train_dir_d, params = list(iterations = 275))

speed_cat_o = catboost.train(learn_pool = cat_train_speed_o, params = list(iterations = 400))
speed_cat_d = catboost.train(learn_pool = cat_train_speed_d, params = list(iterations = 600))

acc_cat_o = catboost.train(learn_pool = cat_train_acc_o, params = list(iterations = 950))
acc_cat_d = catboost.train(learn_pool = cat_train_acc_d, params = list(iterations = 1200))

#save models
catboost.save_model(dir_cat_o, model_path = here("models", "final_models", "offense", "dir.cbm"))
catboost.save_model(dir_cat_d, model_path = here("models", "final_models", "defense", "dir.cbm"))
catboost.save_model(speed_cat_o, model_path = here("models", "final_models", "offense", "speed.cbm"))
catboost.save_model(speed_cat_d, model_path = here("models", "final_models", "defense", "speed.cbm"))
catboost.save_model(acc_cat_o, model_path = here("models", "final_models", "offense", "acc.cbm"))
catboost.save_model(acc_cat_d, model_path = here("models", "final_models", "defense", "acc.cbm"))





