############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(ggrepel)
library(xgboost)

############################################### Description ############################################### 
#' predicting x,y in next frame by first updating (predicting) direction, speed, and acceleration



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
#' 
#' also right now this is only using players_to_predict, not sure if it makes sense to include the other players?


#' first fit model
#' not worried about tuning right now


#modelling the future change in dir, s, a instead of the actual dir, s, a value...
data_mod = train %>% 
  #order the columns
  select(fut_dir_diff, fut_s_diff, fut_a_diff, game_player_play_id, game_play_id, everything()) %>%
  #add estimation to post throw
   mutate(est_dir = ifelse(throw == "pre", dir, est_dir),
          est_speed = ifelse(throw == "pre", s, est_speed),
          est_acc = ifelse(throw == "pre", a, est_acc))  %>%
  #de-select unnecessary feature columns - things that can't be calculated post throw
  select(-c(game_id, nfl_id, play_id, o, player_to_predict, s, a, dir,
          player_birth_date, num_frames_output, num_frames))


#these have the true x,y values but that's ok
#fitting it on true value calculations is ok, but we just can't use the true values to calculate when we actually predict

#' 80% train, 20% test - by game_player_play groups
set.seed(1999)
n_game_player_plays = data_mod %>% pull(game_player_play_id) %>% unique() %>% length() #46,045
split = sample(unique(data_mod$game_player_play_id), size = round(0.8*n_game_player_plays))
curr_train = data_mod %>% filter(game_player_play_id %in% split) %>%
  filter(!is.na(fut_dir_diff) & !is.na(fut_s_diff) & !is.na(fut_a_diff)) #remove NA responses
curr_test = data_mod %>% filter(!(game_player_play_id %in% split))

#filter out the s,a diffs in training set that are clearly impossible
curr_train[,1:3] %>% summary()
curr_train$fut_s_diff %>% quantile(probs = c(0.001, 0.999), na.rm = TRUE)
curr_train$fut_a_diff %>% quantile(probs = c(0.001, 0.999), na.rm = TRUE)
curr_train = data_mod %>% filter(abs(fut_s_diff) <= 1,
                               abs(fut_a_diff) <= 11)

#fit models

#remove all the player-specific stuff for now, just focus on kinematics and ball landing features
xg_train_df = curr_train %>%
  select(-c(game_player_play_id, game_play_id, frame_id, x, y,
            ball_land_x, ball_land_y, player_name, player_height, 
            player_weight, player_role))

dir_xg = xgboost(data =  data.matrix(xg_train_df[,-c(1,2,3)]), 
                 label = xg_train_df$fut_dir_diff,
                 nrounds = 200, print_every_n = 10)

speed_xg = xgboost(data =  data.matrix(xg_train_df[,-c(1,2,3)]), 
                   label = xg_train_df$fut_s_diff,
                   nrounds = 200, print_every_n = 10)

acc_xg = xgboost(data =  data.matrix(xg_train_df[,-c(1,2,3)]), 
                 label = xg_train_df$fut_a_diff,
                 nrounds = 200, print_every_n = 10)

#feature importance
xgb.importance(model = dir_xg)
xgb.importance(model = speed_xg)
xgb.importance(model = acc_xg)

#' double check that the directions are all working properly
#' I swear ball_land_dir_diff should be more important

#just predict on post throw
curr_test_pred = curr_test %>% 
  filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) %>% #filter for post throw only
         #mutate the unknowns to be NA to ensure the model isn't using any future known data
         mutate(across(-c(game_player_play_id, game_play_id, throw, frame_id, play_direction, 
                   absolute_yardline_number, player_name, player_height, player_weight, 
                   player_side, player_role, ball_land_x, ball_land_y, prop_play_complete), 
                ~ ifelse(throw == "pre", .x, NA))) 

#what if you fit the models only on post throw?
#or maybe filter out the first few frames in each play


#attempting to do this in parallel
#with future and furrr package

library(foreach)
library(doParallel)

# Set up cluster
num_cores = parallel::detectCores() - 2
cl = makeCluster(num_cores)
registerDoParallel(cl)

#for testing
#curr_test_pred = curr_test_pred[1:500,]

#storing restuls
#results_pred = list()


#these are what we should parallelize over 
game_player_play_ids = curr_test_pred$game_player_play_id %>% unique()

set.seed(1999)

start = Sys.time()

results = foreach(group_id = game_player_play_ids, .combine = rbind, .packages = c("tidyverse", "xgboost", "doParallel", "progressr")) %dopar% {
  
  #single player on single play
  curr_game_player_play_group = curr_test_pred %>% filter(game_player_play_id == group_id)
  
  #inner_group_results = list()
  
  #loop through frames in play (not in parallel)
  foreach(i = 1:nrow(curr_game_player_play_group), .combine = rbind) %do% {
    curr_row = curr_game_player_play_group[i,]
    
    #initialize position as last observed values before throw
    if (curr_row$throw == "pre") { 
      #if last observation pre throw, predict next frame position using true observed kinematic values
      pred_dist_diff = curr_row$est_speed*0.1 + curr_row$est_acc*0.5*0.1^2
      
      pred_x = curr_row$x + cos(((90 - curr_row$est_dir) %% 360)*pi/180)*pred_dist_diff
      pred_y = curr_row$y + sin(((90 - curr_row$est_dir) %% 360)*pi/180)*pred_dist_diff
      
      #predict dir, s, a
      xg_pred_df = curr_row %>%  #row to predict on
        select(all_of(dir_xg$feature_names))
      
      pred_dir_diff = predict(dir_xg, data.matrix(xg_pred_df))
      pred_s_diff = predict(speed_xg, data.matrix(xg_pred_df))
      pred_a_diff = predict(acc_xg, data.matrix(xg_pred_df))
      
      #predicted dir, s, a
      pred_dir = curr_row$pred_dir = curr_row$est_dir + pred_dir_diff
      pred_s = curr_row$pred_s = curr_row$est_speed + pred_s_diff
      pred_a = curr_row$pred_a = curr_row$est_acc + pred_a_diff
      
      #### Do we predict x,y first, then predict dir, s, a?,
      # if so then we can actually use the predicted x,y as a feature in the dir, s, a model...
      # problem is when we use the true future x,y to train, it might b too good and much better than our predictions
      # making model say the leading x,y is more important than it actually is...
      
      ## could also do this the other way around too, predict the future dir, s, a then predict future x,y
      
      
      # maybe theres a way to do this simultaneously...? like do it both ways, then take the average of each prediction?
      
    } else {
      prev_row = inner_results_pred
      
      #first predict kinematics at current frame using previous frame's predicted position
      curr_row$x = prev_row$pred_x
      curr_row$y = prev_row$pred_y
      
      #set current position and dir, s, a as previous prediction 
      curr_row$est_dir = prev_row$pred_dir
      curr_row$est_speed = prev_row$pred_s 
      curr_row$est_acc = prev_row$pred_a
      
      ### predict kinematics using xg models
      #update all features necessary for dir,s,a models
      prev_curr_frame_df = prev_row %>%
        select(-c(starts_with("pred_"))) %>% #remove the pred columns so we can bind
        rbind(curr_row)
      
      # now get change in kinematics and derived features
      prev_curr_frame_df = prev_curr_frame_df %>%
        change_in_kinematics() %>%
        derived_features() %>%
        mutate(prev_x_diff = x - lag(x),
               prev_y_diff = y - lag(y))
      
      #manually update prop_play_complete
      prev_curr_frame_df$prop_play_complete[2] = curr_row$prop_play_complete
      
      #predict dir, s, a
      xg_pred_df = prev_curr_frame_df[2,] %>%  #row to predict on
        select(all_of(dir_xg$feature_names))
      
      pred_dir_diff = predict(dir_xg, data.matrix(xg_pred_df))
      pred_s_diff = predict(speed_xg, data.matrix(xg_pred_df))
      pred_a_diff = predict(acc_xg, data.matrix(xg_pred_df))
      
      pred_dir = curr_row$est_dir + pred_dir_diff
      pred_s = curr_row$est_speed + pred_s_diff
      pred_a = curr_row$est_acc + pred_a_diff
      
      #update the remaining features that rely on predicted dir, s, a
      
      #finally predict next frame position
      pred_dist_diff = pred_s*0.1 + pred_a*0.5*0.1^2
      
      pred_x = curr_row$x + cos(((90 - pred_dir) %% 360)*pi/180)*pred_dist_diff
      pred_y = curr_row$y + sin(((90 - pred_dir) %% 360)*pi/180)*pred_dist_diff
    }
    
    #store predicted positions
    curr_row$pred_x = pred_x
    curr_row$pred_y = pred_y
    
    #store predicted kinematics
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
results



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

#the true x,y values
true_x_y = train %>%
  filter(!(game_player_play_id %in% split)) %>% 
  rename(true_x = x, true_y = y) %>% 
  select(game_player_play_id, frame_id, true_x, true_y)

#bind results into df
results_pred = results %>%
  bind_rows() %>%
  left_join(true_x_y, by = c("game_player_play_id", "frame_id")) #join true x,y values
 



group_id = 1

results_pred_single_play = results_pred %>% 
  group_by(game_player_play_id) %>%
  filter(cur_group_id() == group_id) %>%
  ungroup() %>%
  select(game_player_play_id, frame_id, x, true_x, y, true_y) %>% 
  rename(pred_x = x, pred_y = y)
results_pred_single_play

plot_player_movement_pred(group_id = unique(results_pred_single_play$game_player_play_id),
                          group_id_preds = results_pred_single_play %>% select(frame_id, pred_x, pred_y))


group_id = 9
#now plot multiple players on same play with predictions
multi_player_pred_single_play = results_pred %>% 
  group_by(game_play_id) %>%
  filter(cur_group_id() == group_id) %>%
  ungroup() %>%
  select(game_player_play_id, game_play_id, frame_id, x, true_x, y, true_y) %>% 
  rename(pred_x = x, pred_y = y)
multi_player_pred_single_play

#multi_player_movement(group_id = unique(multi_player_pred_single_play$game_play_id))
multi_player_movement_pred(group_id = unique(multi_player_pred_single_play$game_play_id),
                           group_id_preds = multi_player_pred_single_play %>% select(game_player_play_id, frame_id, pred_x, pred_y))

#lots of players dont have predictions since they werent in the test set

#' the paths aren't curving to the landing point enough
#' makes me think the ball_land_dir_diff feature is broken or not enough



### RMSE ###

results_rmse = results_pred %>%
  filter(throw == "post") %>%
  group_by(game_player_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))

results_rmse

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



