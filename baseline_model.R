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



#' first fit model on entire dataset
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
            player_weight, player_role, player_position))

dir_xg = xgboost(data =  data.matrix(xg_train_df[,-c(1,2,3)]), 
                 label = xg_train_df$fut_dir_diff,
                 nrounds = 100, print_every_n = 10)

speed_xg = xgboost(data =  data.matrix(xg_train_df[,-c(1,2,3)]), 
                   label = xg_train_df$fut_s_diff,
                   nrounds = 100, print_every_n = 10)

acc_xg = xgboost(data =  data.matrix(xg_train_df[,-c(1,2,3)]), 
                 label = xg_train_df$fut_a_diff,
                 nrounds = 100, print_every_n = 10)

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
                   player_position, player_side, player_role, ball_land_x, ball_land_y, prop_play_complete), 
                ~ ifelse(throw == "pre", .x, NA))) 

#what if you fit the models only on post throw?
#or maybe filter out the first few frames in each play


#attempting to do this in parallel
#with future and furrr package

library(foreach)
library(doParallel)

# Set up cluster
num_cores = 16
cl = makeCluster(num_cores)
registerDoParallel(cl)

#for testing
#curr_test_pred = curr_test_pred[1:1000,]

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
      #set current position and dir, s, a as previous prediction 
      curr_row$x = prev_row$pred_x
      curr_row$y = prev_row$pred_y
      #prev_x = prev_row$x
      #prev_y = prev_row$y
      
      #current dir, s, a
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


group_id = 50
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

#rmse_boxplot + ylim(c(0, 5))

#across entire dataset
results_pred %>% 
  filter(throw == "post") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))

#boxplot of distance of predicted vs true of each frame across all players, games, plays
pred_error = results_pred %>% 
  filter(throw == "post") %>%
  mutate(error_dist = get_dist(x_diff = pred_x - true_x, y_diff = pred_y - true_y)) 

pred_error %>%
  ggplot(mapping = aes(y = error_dist)) +
  geom_boxplot() +
  theme_bw()

#root mean squared dist
pred_error %>% summarise(mean = mean(error_dist)) %>% sqrt()
#why isn't this same as above rmse?




############################################### Predicting on Test Set ############################################### 

test = read.csv(file = here("data", "test_input.csv"))

test = test %>% mutate(player_to_predict = as.logical(player_to_predict), #change player_to_predict to logical
                       player_birth_date = ymd(player_birth_date), #change player_birth_date to date
                       across(where(is.character), as.factor)) #change remaining character types to factors

#make the tidying step above a function so we don't have to redo all of this

#feature creation
test_derived = test %>%
  filter(player_to_predict) %>%
  group_by(game_id, play_id) %>%
  mutate(game_play_id = cur_group_id()) %>% #get game_play_id
  ungroup() %>%
  group_by(game_id, nfl_id, play_id) %>%
  mutate(game_player_play_id = cur_group_id()) %>% #get game_player_play_id
  #first calculate s, a, dir from true values of x,y values
  mutate(prev_x_diff = lead(x, n = lead_frames) - lag(x, n = lag_frames, default = NA), #the difference in x direction from previous frame in yards
         prev_y_diff = lead(y, n = lead_frames) - lag(y, n = lag_frames, default = NA), #the difference in y direction from previous frame in yards
         dist_diff = sqrt(prev_x_diff^2 + prev_y_diff^2), #distance travelled from previous frame in yards
         est_speed = dist_diff/((window_size)/10), #yards/second (1 frame is 0.1 seconds)
         est_acc_vector = (lead(est_speed, n = lead_frames) - lag(est_speed, n = lag_frames))/((window_size)/10), #this has directions (negative accelerations)
         est_acc_scalar = abs(est_acc_vector),
         est_dir = get_dir(x_diff = prev_x_diff, y_diff = prev_y_diff),
         #distance between current x,y and ball land x,y
         ball_land_diff_x = ball_land_x - x,
         ball_land_diff_y = ball_land_y - y,
         prev_s_diff = est_speed - lag(est_speed), #difference in speed from previous to current frame
         prev_a_diff = est_acc_vector - lag(est_acc_vector), #difference in acc from previous to current frame
         fut_s_diff = lead(est_speed) - est_speed, #difference in speed from current to future frame
         fut_a_diff = lead(est_acc_vector) - est_acc_vector #difference in acc from current to future frame
  ) %>%
  #other derived covariates
  mutate(prev_dir_diff_pos = (est_dir - lag(est_dir)) %% 360, #change in dir from previous to current frame - positive direction
         prev_dir_diff_neg = (-(est_dir - lag(est_dir))) %% 360, #change in dir from previous to current frame - negative direction
         fut_dir_diff_pos = (lead(est_dir) - est_dir) %% 360, #change in dir from current to future frame - positive direction
         fut_dir_diff_neg = (-(lead(est_dir) - est_dir)) %% 360, #change in dir from current to future frame - negative direction
         #the direction needed from current point to go to the ball landing point
         curr_ball_land_dir = get_dir(x_diff = ball_land_x - x, y_diff = ball_land_y - y),
         ball_land_dir_diff_pos = (est_dir - curr_ball_land_dir) %% 360,
         ball_land_dir_diff_neg = (-(est_dir - curr_ball_land_dir)) %% 360,
         dist_ball_land = get_dist(x_diff = ball_land_x - x, y_diff = ball_land_y - y), #the distance where the player currently is to where the ball will land
         prop_play_complete = frame_id/max(frame_id), #proportion of play complete - standardizes frame ID
         prop_play_complete_bin = case_when( #bin it into quarters mainly for plotting
           prop_play_complete <= 0.25 ~ "0-0.25",
           prop_play_complete > 0.25 & prop_play_complete <= 0.5 ~ "0.25-0.5",
           prop_play_complete > 0.5 & prop_play_complete <= 0.75 ~ "0.5-0.75",
           prop_play_complete > 0.75 ~ "0.75-1"
         )) %>%
  ungroup() %>%
  rowwise() %>%
  mutate(#vectorized change in direction between previous and current frame
    prev_dir_diff = ifelse(prev_dir_diff_pos <= prev_dir_diff_neg, prev_dir_diff_pos, -prev_dir_diff_neg), 
    #vectorized change in direction between current and future frame
    fut_dir_diff = ifelse(fut_dir_diff_pos <= fut_dir_diff_neg, fut_dir_diff_pos, -fut_dir_diff_neg),
    #difference in current direction of player and direction needed to go to to reach ball land (x,y)
    ball_land_dir_diff = ifelse(ball_land_dir_diff_pos <= ball_land_dir_diff_pos, -ball_land_dir_diff_neg)) %>%
  select(-c(prev_dir_diff_pos, prev_dir_diff_neg, 
            fut_dir_diff_pos, fut_dir_diff_neg, 
            ball_land_dir_diff_pos, ball_land_dir_diff_neg)) %>%
  ungroup() %>%
  mutate(throw = "pre")


#add rows for post throw
stable_features = test_derived %>%
  group_by(game_player_play_id) %>%
  select(game_player_play_id, player_to_predict, num_frames_output, play_direction, absolute_yardline_number, player_name, player_height, 
         player_weight, player_birth_date, player_position, player_side, player_role, ball_land_x, ball_land_y) %>%
  dplyr::slice(1) %>%
  mutate(throw = "post") %>%
  uncount(weights = num_frames_output, .remove = FALSE) #duplicate the rows by num_frames_output

  

#join these back to test_derived
test_pred = test_derived %>%
  full_join(stable_features, by = intersect(colnames(stable_features), colnames(test_derived))) %>%
  group_by(game_player_play_id) %>%
  mutate(frame_id = row_number()) %>% #mutate proper frame_id
  select(-c(game_id, nfl_id, play_id, player_to_predict, player_birth_date, 
            o, num_frames_output, est_speed, est_acc_vector, est_acc_scalar, est_dir)) %>%
  select(fut_dir_diff, fut_s_diff, fut_a_diff, everything()) %>%
  arrange(game_player_play_id, frame_id) %>%
  filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) %>% #filter for post throw only
  ungroup() %>%
  rename(prev_dist_diff = dist_diff)
  


#storing restuls
results_pred_test = list()

#loop
set.seed(1999)
for (i in 1:nrow(test_pred)) {
  if(i %% 10000 == 0) {print(paste0(round(i/nrow(test_pred), 2), " complete"))} #see progress
  
  curr_row = test_pred[i,]
  
  #initialize position as last observed values before throw
  if (curr_row$throw == "pre") { 
    #if last observation pre throw, predict next frame position using true observed kinematic values
    pred_dist_diff = curr_row$s*0.1 + curr_row$a*0.5*0.1^2
    
    pred_x = curr_row$x + cos(((90 - curr_row$dir) %% 360)*pi/180)*pred_dist_diff
    pred_y = curr_row$y + sin(((90 - curr_row$dir) %% 360)*pi/180)*pred_dist_diff
    
    #predict dir, s, a
    xg_pred_df = curr_row %>%  #row to predict on
      select(all_of(dir_xg$feature_names))
    
    #predict change in dir, s, a
    pred_dir_diff = predict(dir_xg, data.matrix(xg_pred_df))
    pred_s_diff = predict(speed_xg, data.matrix(xg_pred_df))
    pred_a_diff = predict(acc_xg, data.matrix(xg_pred_df))
    
    #predicted dir, s, a
    pred_dir = curr_row$pred_dir = curr_row$dir + pred_dir_diff
    pred_s = curr_row$pred_s = curr_row$s + pred_s_diff
    pred_a = curr_row$pred_a = curr_row$a + pred_a_diff
    
    
    
    
    #### Do we predict x,y first, then predict dir, s, a?,
    # if so then we can actually use the predicted x,y as a feature in the dir, s, a model...
    # problem is when we use the true future x,y to train, it might b too good and much better than our predictions
    # making model say the leading x,y is more important than it actually is...
    
    ## could also do this the other way around too, predict the future dir, s, a then predict future x,y
    
    
    # maybe theres a way to do this simultaneously...? like do it both ways, then take the average of each prediction?
    
    
    
  } else {
    prev_row = results_pred_test[[i-1]]
    
    #first predict kinematics at current frame using previous frame's predicted position
    #set current position as previous predicted position
    curr_x = prev_row$pred_x
    curr_y = prev_row$pred_y
    prev_x = prev_row$x
    prev_y = prev_row$y
    
    curr_row$x = curr_x
    curr_row$y = curr_y
    
    ### predict kinematics using xg models
    #update all features necessary for dir,s,a models
    
    prev_x_diff = curr_row$prev_x_diff = curr_x - prev_x
    prev_y_diff = curr_row$prev_y_diff = curr_y - prev_y
    
    #current dir, s, a
    curr_dir = prev_row$pred_dir
    curr_s = prev_row$pred_s 
    curr_a = prev_row$pred_a
    
    curr_row$dir = curr_dir
    curr_row$s = curr_s
    curr_row$a = curr_a
    
    #other features
    prev_dir_diff_pos = (curr_row$dir - prev_row$dir) %% 360 #change in dir from previous to current frame - positive direction
    prev_dir_diff_neg = (-(curr_row$dir - prev_row$dir)) %% 360 #change in dir from previous to current frame - negative direction
    curr_row$prev_dir_diff = ifelse(prev_dir_diff_pos <= prev_dir_diff_neg, prev_dir_diff_pos, -prev_dir_diff_neg)
    
    curr_row$prev_s_diff = curr_s - prev_row$s
    curr_row$prev_a_diff = curr_a - prev_row$a
    
    curr_row$prev_dist_diff = get_dist(x_diff = prev_x_diff, y_diff = prev_y_diff)
    curr_row$curr_ball_land_dir = get_dir(x_diff = curr_row$ball_land_x - curr_x, y_diff = curr_row$ball_land_y - curr_y)
    curr_row$dist_ball_land = get_dist(x_diff = curr_row$ball_land_x - curr_x, y_diff = curr_row$ball_land_y - curr_y)
    
    ball_land_dir_diff_pos = (curr_row$dir - curr_row$curr_ball_land_dir) %% 360
    ball_land_dir_diff_neg = (-(curr_row$dir - curr_row$curr_ball_land_dir)) %% 360
    
    curr_row$ball_land_dir_diff = ifelse(ball_land_dir_diff_pos <= ball_land_dir_diff_pos, -ball_land_dir_diff_neg)
    
    curr_row$ball_land_diff_x = curr_row$ball_land_x - curr_x
    curr_row$ball_land_diff_y = curr_row$ball_land_y - curr_y
    
    
    #predict dir, s, a
    xg_pred_df = curr_row %>%  #row to predict on
      select(all_of(dir_xg$feature_names))
    
    pred_dir_diff = predict(dir_xg, data.matrix(xg_pred_df))
    pred_s_diff = predict(speed_xg, data.matrix(xg_pred_df))
    pred_a_diff = predict(acc_xg, data.matrix(xg_pred_df))
    
    pred_dir = curr_dir + pred_dir_diff
    pred_s = curr_s + pred_s_diff
    pred_a = curr_a + pred_a_diff
    
    #update the remaining features that rely on predicted dir, s, a
    
    #finally predict next frame position
    pred_dist_diff = pred_s*0.1 + pred_a*0.5*0.1^2
    
    pred_x = curr_x + cos(((90 - pred_dir) %% 360)*pi/180)*pred_dist_diff
    pred_y = curr_y + sin(((90 - pred_dir) %% 360)*pi/180)*pred_dist_diff
  }
  
  #store predicted positions
  curr_row$pred_x = pred_x
  curr_row$pred_y = pred_y
  
  #store predicted kinematics
  curr_row$pred_dir = pred_dir
  curr_row$pred_s = pred_s
  curr_row$pred_a = pred_a
  
  #store
  results_pred_test[[i]] = curr_row
}
#def need to do this in parallel

#bind results into df
results_pred_test = results_pred %>%
  bind_rows() 


results_pred_test







############################################### Tuning - Cross Validation ############################################### 



#' use cross-validation by splitting up different game-player-plays groups
num_folds = 10


#' do this in parallel
library(foreach)
library(doParallel)

n_cores = 10
cluster = parallel::makeCluster(
  n_cores, 
  type = "PSOCK"
)
doParallel::registerDoParallel(cl = cluster)

set.seed(1999)
# results = foreach(folds = 1:num_folds) %dopar% {
#   curr_cv_split = (sample(nrow(train)) %% folds) + 1 #indeces of the current cv split
#   
#   curr_train = train_sub[(curr_cv_split != fold),] #current training set
#   curr_test = train_sub[(curr_cv_split == fold),] #current test set
#   
#   #current model fit on training set
#   curr_xg = xgboost(data = data.matrix(curr_train[,-1]), label = curr_train$accident_risk,
#                     max.depth = 3, eta = 0.39, nrounds = 100, objective = "reg:squarederror",
#                     min_child_weight = 1,  colsample_bytree = colsample_bytree, subsample = subsample)
#   curr_rmse = sqrt(mean((curr_test$accident_risk - predict(curr_xg, data.matrix(curr_test[,-1])))^2))
#   curr_rmse #return RMSE
# }

set.seed(1999)
results = list()
for (fold in 1:num_folds) {
  print(fold) #see progress
  
  #indeces of the current cv split
  curr_cv_split = (sample(n_game_player_plays) %% num_folds) + 1 
  #ordering the response columns first and removing any NAs
  data_mod = train %>% 
    #order the columns
    select(est_dir, est_speed, est_acc_vector, game_player_play_id, everything()) %>%
    #remove NA responses
    filter(!is.na(est_dir) & !is.na(est_speed) & !is.na(est_acc_vector)) %>%
    #unselect unnecessary feature columns
    select(-c(game_id, nfl_id, play_id, s, a, dir, player_to_predict, player_birth_date,
              est_acc_scalar))
  
  #current training set
  curr_train = data_mod %>% filter(game_player_play_id %in% which(curr_cv_split != fold))
  #current test set
  curr_test = data_mod %>% filter(game_player_play_id %in% which(curr_cv_split == fold) )
  
  
  #now fit a direction, speed, and acceleration model
  curr_dir_xg = xgboost(data = data.matrix(curr_train[,-c(1, 4)]), label = curr_train$est_dir,
                        nrounds = 50, print_every_n = 10)
  curr_speed_xg = xgboost(data = data.matrix(curr_train[,-c(2, 4)]), label = curr_train$est_speed,
                          nrounds = 50, print_every_n = 10)
  curr_acc_xg = xgboost(data = data.matrix(curr_train[,-c(3, 4)]), label = curr_train$est_acc_vector,
                        nrounds = 50, print_every_n = 10)
  # #store predictions
  # results[[fold]] = curr_test %>%
  #   mutate(dir_pred = predict(curr_dir_xg, data.matrix(curr_test[,-c(1, 4)])),
  #          speed_pred = predict(curr_speed_xg, data.matrix(curr_test[,-c(2, 4)])),
  #          acc_pred = predict(curr_acc_xg, data.matrix(curr_test[,-c(3, 4)])))
}
#' we don't predict here
#' need to do it later on sequentially after we have prediction of future frame
#' 
#' if we're not worried about tuning here, just set it to one fold and train on entire set

data_mod = train %>% 
  #order the columns
  select(est_dir, est_speed, est_acc_vector, game_player_play_id, everything()) %>%
  #remove NA responses
  filter(!is.na(est_dir) & !is.na(est_speed) & !is.na(est_acc_vector),
         player_to_predict) %>%
  #unselect unnecessary feature columns
  select(-c(game_id, nfl_id, play_id, s, a, dir, player_to_predict, player_birth_date,
            est_acc_scalar))


results_pred = results %>% 
  bind_rows() %>%
  filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) %>% #filter for post throw only
  #take the mean prediction if plays were in multiple folds
  group_by(game_player_play_id, frame_id) %>%
  mutate(dir_pred = mean(dir_pred),
         speed_pred = mean(speed_pred),
         acc_pred = mean(acc_pred)) %>%
  dplyr::slice(1) %>%
  ungroup() %>%
  #update position
  mutate(pred_dist_diff = speed_pred*0.1 + acc_pred*0.5*0.1^2)


preds = matrix(ncol = 2, 
               nrow = nrow(results_pred),
               dimnames = list(seq(1:nrow(results_pred)), c("x", "y")))

#add predictions back to results df
results_pred$pred_x = preds[,1]
results_pred$pred_y = preds[,2]


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



