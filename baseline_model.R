############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(ggrepel)
library(xgboost)
library(catboost)

############################################### Description ############################################### 
#' predicting x,y from direction, speed, acceleration in current frame and updating (predicting) direction, speed, and acceleration in next frame
#' three models that predict change in dir, s, a each
#' separate dir, s, a models for offense and defense 


############################################### Baseline Model ############################################### 

#import data
train = read.csv(file = here("data", "train_clean.csv"))

source(here("helper.R"))

#' end model is a simple kinematics formula:
#' pos_next = pos_curr + vel_curr*0.1 + 0.5*acc_curr*(0.1^2) - 0.1 seconds between frames
#' this is a straight line in the current direction (dir_curr)
#' this is a simple model that can then be built upon
#' autoregressive, predicts position in next frame from current frame
#' need three seperate models: direction, speed, acceleration
#' once you have these three things, then its a simple kinematics formula

#' ignore the player info and everything else right now, just keep it simple
#' also right now this is only using players_to_predict, not sure if it makes sense to include the other players?


#' first fit model
#' not worried about tuning right now

#modelling the future change in dir, s, a instead of the actual dir, s, a value...
data_mod = train %>% 
  #order the columns
  select(fut_dir_diff, fut_s_diff, fut_a_diff, game_player_play_id, game_play_id, everything()) %>%
  #add estimation to post throw
   mutate(est_dir = ifelse(throw == "pre", dir, est_dir), #use recorded dir, s if possible over estimated...
          est_speed = ifelse(throw == "pre", s, est_speed))  %>% #do not use recorded a since its a scalar, we want vector
  #de-select unnecessary feature columns - things that can't be calculated post throw
  select(-c(game_id, nfl_id, play_id, o, player_to_predict, s, a, dir,
          player_birth_date, num_frames_output, num_frames)) %>%
  mutate(across(where(is.character), as.factor))

#what proportion of play being complete is ball thrown
data_mod %>% filter(throw == "post" & lag(throw) == "pre") %>% pull(prop_play_complete) %>% hist()

#min is 0.225 - fit models mostly on post-throw

#these have the true x,y values but that's ok
#fitting it on true value calculations is ok, but we just can't use the true values to calculate when we actually predict

#' 80% train, 20% test - by game_player_play groups
set.seed(1999)
n_game_player_plays = data_mod %>% pull(game_player_play_id) %>% unique() %>% length() #46,045
split = sample(unique(data_mod$game_player_play_id), size = round(0.8*n_game_player_plays))
curr_train = data_mod %>% filter(game_player_play_id %in% split) %>%
  filter(!is.na(fut_dir_diff) & !is.na(fut_s_diff) & !is.na(fut_a_diff)) #remove NA responses
curr_test = data_mod %>% filter(!(game_player_play_id %in% split))

#filter out the dir,s,a diffs in training set that are clearly impossible
curr_train[,1:3] %>% summary()
curr_train$fut_s_diff %>% quantile(probs = c(0.001, 0.999), na.rm = TRUE)
curr_train$fut_a_diff %>% quantile(probs = c(0.001, 0.999), na.rm = TRUE)
curr_train = data_mod %>% filter(abs(fut_s_diff) <= 1,
                               abs(fut_a_diff) <= 11) %>%
  filter(prop_play_complete >= 0.4 | throw == "post") #play is more than 10% complete


#fit models
#remove all the player-specific stuff for now, just focus on kinematics and ball landing features
cat_train_df = curr_train %>%
  select(-c(game_player_play_id, game_play_id, closest_player_id, closest_player_x, closest_player_y,
            frame_id, x, y, ball_land_x, ball_land_y, player_name, player_height, player_weight, player_role))

#offense and defense training sets
train_o = cat_train_df %>% filter(player_side == "Offense") %>% select(-c(player_side, fut_dir_diff, fut_s_diff, fut_a_diff))
train_d = cat_train_df %>% filter(player_side == "Defense") %>% select(-c(player_side, fut_dir_diff, fut_s_diff, fut_a_diff))
train_o_labels = cat_train_df %>% filter(player_side == "Offense") %>% select(fut_dir_diff, fut_s_diff, fut_a_diff)
train_d_labels = cat_train_df %>% filter(player_side == "Defense") %>% select(fut_dir_diff, fut_s_diff, fut_a_diff)

#df in right type for catboost
cat_train_dir_o = catboost.load_pool(train_o, label = train_o_labels$fut_dir_diff)
cat_train_dir_d = catboost.load_pool(train_d, label = train_d_labels$fut_dir_diff)
cat_train_speed_o = catboost.load_pool(train_o, label = train_o_labels$fut_s_diff)
cat_train_speed_d = catboost.load_pool(train_d, label = train_d_labels$fut_s_diff)
cat_train_acc_o = catboost.load_pool(train_o, label = train_o_labels$fut_a_diff)
cat_train_acc_d = catboost.load_pool(train_d, label = train_d_labels$fut_a_diff)


#fit
dir_cat_o = catboost.train(cat_train_dir_o, params = list(metric_period = 50))
dir_cat_d = catboost.train(cat_train_dir_d, params = list(metric_period = 50))
#can optionally use a test set here and it automatically evaluates???
speed_cat_o = catboost.train(cat_train_speed_o, params = list(metric_period = 50))
speed_cat_d = catboost.train(cat_train_speed_d, params = list(metric_period = 50))

acc_cat_o = catboost.train(cat_train_acc_o, params = list(metric_period = 50))
acc_cat_d = catboost.train(cat_train_acc_d, params = list(metric_period = 50))


#just predict on post throw
curr_test_pred = curr_test %>% 
  filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) %>% #filter for post throw only
         #mutate the unknowns to be NA to ensure the model isn't using any future known data
         mutate(across(-c(game_player_play_id, game_play_id, throw, frame_id, play_direction, 
                   absolute_yardline_number, player_name, player_height, player_weight, player_position,
                   player_side, player_role, ball_land_x, ball_land_y, prop_play_complete), 
                ~ ifelse(throw == "pre", .x, NA))) 

#attempting to do this in parallel
#with future and furrr package

library(foreach)
library(doParallel)

# Set up cluster
num_cores = parallel::detectCores() - 2
cl = makeCluster(num_cores)
registerDoParallel(cl)

#for testing
curr_test_pred = curr_test_pred[1:100,]

#these are what we should parallelize over 
game_play_ids = curr_test_pred$game_play_id %>% unique()

#this is my predict() function

set.seed(1999)
start = Sys.time()
results = foreach(group_id = game_play_ids, .combine = rbind, .packages = c("tidyverse", "doParallel", "catboost")) %do% {
  
  #single player on single play
  curr_game_play_group = curr_test_pred %>% filter(game_play_id == group_id)
  
  #loop through frames in play (not in parallel)
  foreach(i = 1:nrow(curr_game_play_group), .combine = rbind) %do% {
    #in order to incorporate closest player metrics
    #need to simultaneously predict every player in the play on the current frame
    #then you can get the distances and directions to each other from the closest_player_dist_dir function
    
    #so instead of curr_row, we should do curr_frame
    
    
    curr_row = curr_game_play_group[i,]
    
    #initialize position as last observed values before throw
    if (curr_row$throw == "pre") { 
      #if last observation pre throw, predict next frame position using true observed kinematic values
      pred_dist_diff = curr_row$est_speed*0.1 + curr_row$est_acc*0.5*0.1^2
      
      pred_x = curr_row$x + cos(((90 - curr_row$est_dir) %% 360)*pi/180)*pred_dist_diff
      pred_y = curr_row$y + sin(((90 - curr_row$est_dir) %% 360)*pi/180)*pred_dist_diff
      
      #predict future dir, s, a (next frame)
      cat_pred_df = curr_row %>%  #row to predict on
        select(all_of(rownames(dir_cat_d$feature_importances))) %>%
        catboost.load_pool()
      
    } else {
      prev_row = inner_results_pred
      
      #set current position and dir, s, a as previous prediction 
      curr_row$x = prev_row$pred_x
      curr_row$y = prev_row$pred_y
      
      curr_row$est_dir = prev_row$pred_dir
      curr_row$est_speed = prev_row$pred_s 
      curr_row$est_acc = prev_row$pred_a
      
      #predict next frame x,y using current dir, s, a
      pred_dist_diff = curr_row$est_speed*0.1 + curr_row$est_acc*0.5*0.1^2
      pred_x = curr_row$x + cos(((90 - curr_row$est_dir) %% 360)*pi/180)*pred_dist_diff
      pred_y = curr_row$y + sin(((90 - curr_row$est_dir) %% 360)*pi/180)*pred_dist_diff
      
      
      #update features for predicting next dir, s, a 
      prev_curr_frame_df = prev_row %>%
        select(-c(starts_with("pred_"))) %>% #remove the pred columns so we can bind
        rbind(curr_row)
      
      # now get change in kinematics and derived features
      prev_curr_frame_df = prev_curr_frame_df %>%
        change_in_kinematics() %>%
        closest_player_dist_dir() %>%
        derived_features() %>%
        mutate(prev_x_diff = x - lag(x),
               prev_y_diff = y - lag(y))
      
      #predict future dir, s, a (next frame)
      cat_pred_df = prev_curr_frame_df[2,] %>%  #row to predict on
        select(all_of(rownames(dir_cat_d$feature_importances))) %>%
        catboost.load_pool()
    }
    
    fut_dir_diff = ifelse(curr_row$player_side == "Offense", 
                          catboost.predict(dir_cat_o, cat_pred_df),
                          catboost.predict(dir_cat_d, cat_pred_df))
    fut_s_diff = ifelse(curr_row$player_side == "Offense", 
                        catboost.predict(speed_cat_o, cat_pred_df),
                        catboost.predict(speed_cat_d, cat_pred_df))
    fut_a_diff = ifelse(curr_row$player_side == "Offense", 
                        catboost.predict(acc_cat_o, cat_pred_df),
                        catboost.predict(acc_cat_d, cat_pred_df))
    
    #predicted dir, s, a using xg models
    pred_dir = curr_row$pred_dir = curr_row$est_dir + fut_dir_diff
    pred_s = curr_row$pred_s = curr_row$est_speed + fut_s_diff
    pred_a = curr_row$pred_a = curr_row$est_acc + fut_a_diff
    
    #store predicted positions and kinematics
    curr_row$pred_x = pred_x
    curr_row$pred_y = pred_y
    curr_row$pred_dir = pred_dir
    curr_row$pred_s = pred_s
    curr_row$pred_a = pred_a
    
    #return result
    inner_results_pred = curr_row
    inner_results_pred
  }
}
end = Sys.time()
end - start
#results

### experiment with this - figure out the best method - compare rmse at the end
#' 1. predict x,y first, then use future x,y to predict future dir, s, a
#' 2. predict dir,s,a first, then use future dir,s,a to predict future x,y
#' 3. do both and take the average prediciton - do this iteratively, take the mean at every step...

### another thing that would probably improve predictions:
#' generate first predictions using only lag x,y 
#' after you get the predictions, use those to get better estimates of dir, s, a on the current frame
#' by calculating dir,s,a over prev_frame -> leading frame, this gives esimate of kinematics at the actual current frame
#' 
#' can keep doing this until convergence... 
#' but how computationally feasible is this
#' 
#' 


#the true x,y,dir,s,a values
true_vals = train %>%
  filter(!(game_player_play_id %in% split)) %>% 
  rename(true_dir = est_dir, true_s = est_speed, true_a = est_acc, true_x = x, true_y = y) %>% 
  select(game_player_play_id, frame_id, true_dir, true_s, true_a, true_x, true_y)

#bind results into df
results_pred = results %>%
  bind_rows() %>%
  left_join(true_vals, by = c("game_player_play_id", "frame_id")) #join true x,y values


#pred dir, s, a vs true dir, s, a
group_id = 2
dir_s_a_eval(group_id)

#single player movement
curr_game_player_play_id = results_pred %>% 
  group_by(game_player_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_player_play_id) %>% unique()
#curr_game_player_play_id = 32291  

plot_player_movement_pred(group_id = curr_game_player_play_id,
                          group_id_preds = results_pred %>% 
                            filter(game_player_play_id == curr_game_player_play_id) %>%
                            select(frame_id, x, y) %>%
                            rename(pred_x = x, pred_y = y))


#multiple players on play
group_id = 7
curr_game_play_id = results_pred %>% 
  group_by(game_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_play_id) %>% unique()

multi_player_movement_pred(group_id = curr_game_play_id,
                           group_id_preds = results_pred %>%
                             filter(game_play_id == curr_game_play_id) %>%
                             select(game_player_play_id, frame_id, x, y) %>%
                             rename(pred_x = x, pred_y = y))

#lots of players dont have predictions since they weren't in the test set


### RMSE ###

results_rmse = results_pred %>%
  filter(throw == "post") %>%
  group_by(game_player_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
results_rmse %>% arrange(desc(rmse))


#plot
rmse_boxplot = results_rmse %>% 
  ggplot(mapping = aes(y = rmse)) +
  geom_boxplot() +
  theme_bw()
rmse_boxplot

rmse_boxplot + ylim(c(0, 5))

#across entire dataset
results_pred %>% 
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
#catboost off/def models, 0.4 prop_play - 0.834




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



