############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(ggrepel)
library(catboost)
library(scattermore)
library(foreach)
library(doFuture)
library(progressr)
source(here("helper.R"))


############################################### Description ############################################### 
#' predicting x,y from direction, speed, acceleration in current frame and updating (predicting) direction, speed, and acceleration in next frame
#' three models that predict change in dir, s, a each
#' separate dir, s, a models for offense and defense 


############################################### Import data ###############################################

train = read.csv(file = here("data", "train_clean.csv"))
data_mod = read.csv(file = here("data", "data_mod.csv")) %>%
  mutate(across(where(is.character), as.factor)) #for catboost


############################################### Start CV ###############################################

set.seed(1999)
num_folds = 5 #80% train, 20% test
game_play_ids = data_mod %>% pull(game_play_id) %>% unique() %>% sort()
num_plays = length(game_play_ids) #14,107 plays

#the cv splits
split = (sample(game_play_ids) %% num_folds) + 1

#make it a function
#df is data_train_mod - the cleaned dataset to fit models on
#pred_subset is the number of rows to predict on in the test folds - to make this run quicker
cv = function(df, pred_subset = FALSE) {
  
  #set up parallel and progress
  handlers(global = TRUE)
  handlers("progress")
  
  registerDoFuture()
  plan(multisession, workers = 14) #out of 20 cores
  
  
  #loop through folds
  foreach(fold = 1:num_folds, .combine = rbind, .packages = c("tidyverse", "doParallel", "catboost")) %do% {
    train_plays = game_play_ids[split != fold]
    test_plays = game_play_ids[split == fold]
    
    #split df
    data_mod_train = data_mod %>% 
      filter(game_play_id %in% train_plays,
             abs(fut_s_diff) <= 1, #filter out the crazy speeds/accs for training
             abs(fut_a_diff) <= 6)
    data_mod_test = data_mod %>% 
      filter(game_play_id %in% test_plays)
    
    
    ### fit models
    
    #drop unnecessary features
    unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role",
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
    
    #df in right type for catboost
    #train sets
    cat_train_dir_o = catboost.load_pool(train_o, label = train_o_labels$fut_dir_diff)
    cat_train_dir_d = catboost.load_pool(train_d, label = train_d_labels$fut_dir_diff)
    cat_train_speed_o = catboost.load_pool(train_o, label = log(train_o_labels$fut_s)) #try log transform since speed is strictly positive
    cat_train_speed_d = catboost.load_pool(train_d, label = log(train_d_labels$fut_s)) #try log transform since speed is strictly positive
    cat_train_acc_o = catboost.load_pool(train_o, label = train_o_labels$fut_a_diff)
    cat_train_acc_d = catboost.load_pool(train_d, label = train_d_labels$fut_a_diff)
    #test sets
    cat_test_dir_o = catboost.load_pool(test_o, label = test_o_labels$fut_dir_diff)
    cat_test_dir_d = catboost.load_pool(test_d, label = test_d_labels$fut_dir_diff)
    cat_test_speed_o = catboost.load_pool(test_o, label = log(test_o_labels$fut_s)) #try log transform since speed is strictly positive
    cat_test_speed_d = catboost.load_pool(test_d, label = log(test_d_labels$fut_s)) #try log transform since speed is strictly positive
    cat_test_acc_o = catboost.load_pool(test_o, label = test_o_labels$fut_a_diff)
    cat_test_acc_d = catboost.load_pool(test_d, label = test_d_labels$fut_a_diff)
    
    #fit
    #just use basic quick tuning parameters now
    dir_cat_o = catboost.train(learn_pool = cat_train_dir_o, test_pool = cat_test_dir_o, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100)) #num iterations to go past min test error before stop
    dir_cat_d = catboost.train(learn_pool = cat_train_dir_d, test_pool = cat_test_dir_d, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    
    speed_cat_o = catboost.train(learn_pool = cat_train_speed_o, test_pool = cat_test_speed_o, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                             od_type = "Iter", od_wait = 100))
    speed_cat_d = catboost.train(learn_pool = cat_train_speed_d, test_pool = cat_test_speed_d, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                             od_type = "Iter", od_wait = 100))
    #calculate residual variance for log-bias correction
    # o_speed_correction = var(log(test_o_labels$fut_s) - catboost.predict(model = speed_cat_o, pool = cat_test_speed_o))
    # d_speed_correction = var(log(test_d_labels$fut_s) - catboost.predict(model = speed_cat_d, pool = cat_test_speed_d))
    #seems to make predictions worse
    
    acc_cat_o = catboost.train(learn_pool = cat_train_acc_o, test_pool = cat_test_acc_o, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    acc_cat_d = catboost.train(learn_pool = cat_train_acc_d, test_pool = cat_test_acc_d, params = list(iterations = 1000, logging_level = "Silent",
                                                                                                       od_type = "Iter", od_wait = 100))
    
    # #feature importance
    # catboost.get_feature_importance(dir_cat_o)
    # catboost.get_feature_importance(dir_cat_d)
    # catboost.get_feature_importance(speed_cat_o)
    # catboost.get_feature_importance(speed_cat_d)
    # catboost.get_feature_importance(acc_cat_o)
    # catboost.get_feature_importance(acc_cat_d)
    #' experiment with:
    #'  1. shrinking model
    #'  2. drop unused features
    #'  
    #'  see if any of these improve performance
    
    #df to generate predictions on
    data_mod_test_pred = data_mod_test %>% 
      filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) %>% #filter for post throw only
      #mutate the unknowns to be NA to ensure the model isn't using any future known data
      mutate(across(-c(game_player_play_id, game_play_id, throw, frame_id, play_direction, 
                       absolute_yardline_number, player_name, player_height, player_weight, player_position,
                       player_side, player_role, ball_land_x, ball_land_y, prop_play_complete), 
                    ~ ifelse(throw == "pre", .x, NA))) 
    
    
    #these are what we should parallelize over 
    pred_play_ids = data_mod_test_pred$game_play_id %>% unique() %>% sort()
    
    #predict on subset of test folds
    if(is.numeric(pred_subset)) {
      pred_play_ids = pred_play_ids[1:pred_subset]
    }
    
    #' flow of this:
    #'  -loop through all the plays
    #'    -loop through all the frames in a play
    #'    -predict next x,y
    #'    -derive features necessary for next dir, s, a prediction
    #'      -loop through the players in the play "post throw" (player_to_predict == TRUE)
    #'        -predict next dir, s, a
  
    with_progress({
      p = progressor(steps = length(pred_play_ids)) #progress
      
      results = foreach(group_id = pred_play_ids, .combine = rbind, .packages = c("tidyverse", "doParallel", "catboost")) %dopar% {
        p(sprintf("Iteration %d", group_id)) #progress 
        
        #single player on single play
        curr_game_play_group = data_mod_test_pred %>% filter(game_play_id == group_id)
        #the frames in the play
        frames = curr_game_play_group$frame_id %>% unique()
        #player ids in the play we need to predict
        player_ids = curr_game_play_group$game_player_play_id %>% unique()
        
        #loop through frames in play (not in parallel)
        foreach(frame = frames, .combine = rbind) %do% {
          
          #info for all players in current frame
          curr_frame_all_players = data_mod_test_pred %>% filter(game_play_id == group_id, frame_id == frame)
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
              select(-starts_with("closest_")) %>% #deselect the closest player columns so we can merge the right ones
              full_join(closest_player_features, by = c("game_player_play_id")) #%>% #join the min dist/dir
            #mutate(closest_player_dir_diff = min_pos_neg_dir(est_dir - closest_player_dir)) %>%
            #select(-closest_player_dir) #deselect the actual direction
            
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
            cat_pred_df = curr_frame_all_players %>%
              filter(game_player_play_id == player) %>%
              select(all_of(rownames(dir_cat_d$feature_importances))) %>% #pred df for catboost
              catboost.load_pool()
            
            #return row with predicted dir, s, a
            curr_frame_player = curr_frame_all_players %>% 
              filter(game_player_play_id == player) %>% #predict on single player
              mutate(pred_dir = est_dir + ifelse(player_side == "Offense", 
                                                 catboost.predict(dir_cat_o, cat_pred_df), 
                                                 catboost.predict(dir_cat_d, cat_pred_df)),
                     pred_s = ifelse(player_side == "Offense", 
                                     catboost.predict(speed_cat_o, cat_pred_df), 
                                     catboost.predict(speed_cat_d, cat_pred_df)),
                     pred_s = exp(pred_s),
                     pred_a = est_acc + ifelse(player_side == "Offense", 
                                               catboost.predict(acc_cat_o, cat_pred_df), 
                                               catboost.predict(acc_cat_d, cat_pred_df)))
            
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
    
    print(paste0("Fold ", fold, ", (", 100*(fold/num_folds), "% folds done)")) #see progress
  }
}

#run CV
start = Sys.time()
results = cv(data_mod) 
end = Sys.time()
end - start

#stop parallel workers
plan(sequential)

#1000 iterations training, entire dataset takes ___


# if were running the same folds everytime, then just train the catboost models once so you dont have to do it every time
# this will speed it up a lot



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



#the true x,y,dir,s,a values
true_vals = data_mod_test %>% 
  rename(true_dir = est_dir, true_s = est_speed, true_a = est_acc, true_x = x, true_y = y) %>% 
  select(game_player_play_id, frame_id, true_dir, true_s, true_a, true_x, true_y)

#bind results into df
results = results %>%
  left_join(true_vals, by = c("game_player_play_id", "frame_id")) %>% #join true x,y values
  arrange(game_play_id, game_player_play_id, frame_id)


#evaluate dir, s, a results
results %>% select(pred_dir, true_dir) %>% summary()
results %>% select(pred_dir, true_dir) %>% mutate(pred_dir = pred_dir %% 360) %>% summary()
results %>% select(pred_s, true_s) %>% summary()
results %>% select(pred_a, true_a) %>% summary()

#I think we need to mod 360 the direction?
#also some predicted speeds are negative, clearly this is impossible - when this is the case, convert it to 0?
#is there a way for catboost to constrain the response?


#true vs predicted dir
ggplot(data = results, mapping = aes(x = true_dir, y = min_pos_neg_dir(pred_dir - true_dir))) +
  geom_scattermore() +
  xlim(c(0, 360)) + 
  theme_bw() +
  labs(x = "True Direction", y = "Predicted Direction Minus True Direction")

#true vs predicted speed
ggplot(data = results, mapping = aes(x = true_s, y = pred_s)) +
  geom_scattermore() +
  xlim(c(min(c(results$pred_s, results$true_s))), 
       max(c(results$pred_s, results$true_s))) + 
  ylim(c(min(c(results$pred_s, results$true_s))), 
       max(c(results$pred_s, results$true_s))) +
  theme_bw() +
  labs(x = "True Speed", y = "Predicted Speed")

#true vs predicted acc
acc = ggplot(data = results, mapping = aes(x = true_a, y = pred_a)) +
  geom_scattermore() +
  xlim(c(min(c(results$pred_a, results$true_a))), 
       max(c(results$pred_a, results$true_a))) + 
  ylim(c(min(c(results$pred_a, results$true_a))), 
       max(c(results$pred_a, results$true_a))) +
  theme_bw() +
  labs(x = "True Acceleration", y = "Predicted Acceleration")
acc
acc + xlim(c(-10, 10)) + ylim(c(-10,10))
#acceleration seems squashed a bit, maybe include higher true vals in training set?



#pred dir, s, a vs true dir, s, a
group_id = 1
dir_s_a_eval(group_id)

#single player movement
curr_game_player_play_id = results %>% 
  group_by(game_player_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_player_play_id) %>% unique()
#curr_game_player_play_id = 1

plot_player_movement_pred(group_id = curr_game_player_play_id,
                          group_id_preds = results %>% 
                            filter(game_player_play_id == curr_game_player_play_id) %>%
                            select(frame_id, x, y) %>%
                            rename(pred_x = x, pred_y = y))


#multiple players on play
group_id = 1
curr_game_play_id = results %>% 
  group_by(game_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_play_id) %>% unique()
#curr_game_play_id = 2156

multi_player_movement_pred(group_id = curr_game_play_id,
                           group_id_preds = results %>%
                             filter(game_play_id == curr_game_play_id) %>%
                             select(game_player_play_id, frame_id, x, y) %>%
                             rename(pred_x = x, pred_y = y))

#lots of players dont have predictions since they weren't in the test set


### RMSE ###

#worst player_plays
results_rmse_player = results %>%
  filter(throw == "post") %>%
  group_by(game_player_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
results_rmse_player %>% arrange(desc(rmse))

#worst plays
results_rmse_play = results %>%
  filter(throw == "post") %>%
  group_by(game_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
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
results %>% 
  filter(throw == "post") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
#remember this is only 20% of the dataset


#fit on all -                     - 1.111


#fit on prop_play_complete > 0.4  - 1.083

#fit on prop_play_complete > 0.4  - 1.035
#    and separate off,def models 


#fit on throw == "post"           - 1.154


#cat boost off/def models and 0.4 prop_play  - 0.951

#Fixed acceleration!
#catboost off/def models, 0.4 prop_play -  0.834
#above but added closest player features - 0.821

#same as above but filtering out weird data - 0.730
#same as above but band-aid fix to negative speeds - 0.729


#0.711


#offense across entire dataset
results_pred %>% 
  filter(player_side == "Offense", throw == "post") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))

#defense across entire dataset
results_pred %>% 
  filter(player_side == "Defense", throw == "post") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))

#defense is much harder to predict 

#' Off - 0.537
#' Def - 0.929
#' 
#' 
#' 
#'  YOU NEED TO DO CV - ALL THESE RMSEs MIGHT BE USING DIFFERENT TEST SETS



############################################### Misc Ideas I'll get to eventually ############################################### 

#' 1. check if theres anything in test thats not in train... (any player for eg)
#'      there will be

#' 3. figure out how to incorporate orientation, cannot estimate from x,y - they use sensors in player's shoulder pads
#'      maybe estimate it? but how helpful will this even be?

#' 4. make a model to predict direction - then use that
#'      train model to predict direction, 
#'      if a player is turning for eg, the direction isn't just a straight line between current and previous frame, its going to keep curving
#'      this depends on speed, acc, and where the ball is landing
#'      two solutions to this, the gibbs way as before, or fit a model to predict direction

#' 5. certain players are dominant to one side, eg always like to cut right, use this info?


#' 6. another covariate could be the second derivative of the player's curve over the past 5 frames for eg, the sharper the curve, the slower the speed/acc...
#' 
#' 7. create a DAG
#' 
#' 8. Create different models for offense and defense?
#'       defence needs different features, ie, dist/dir to offensive player, or dist/dir to other player...



