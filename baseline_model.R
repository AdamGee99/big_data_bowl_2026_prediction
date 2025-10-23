############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(ggrepel)
library(xgboost)

############################################### TO DO ############################################### 

#' Steps
#' 1. Generate end model
#'      -by end model, I mean final in the line of models to generate the position predictions
#'      -this model simply predicts the player's position on the next frame using direction, speed, acceleration of current frame
#'      -this is an exact formula, no estimation...(simple kinematics formula)
#'      -just use the true observed direction, speed, acceleration values for now, don't worry about estimating them
#'      -see how well this matches up with true observed values
#'      -figure out what features are needed here (is direction, speed, acceleration all or is more neeeded? orientation?)
#' 
#' 
#' 2. start developing models for the each features needed above (direction, speed, acceleration)



############################################### Step 1 ############################################### 

#import data
train = read.csv(file = here("data", "train_clean.csv"))

source(here("helper.R"))

#' end model is a simple kinematics formula:
#' pos_next = pos_curr + vel_curr*0.1 + 0.5*acc_curr*(0.1^2) - 0.1 seconds between frames
#' this is a straight line in the current direction (dir_curr)


#' right now use true s, a, dir to predict next frame
#' if you know the player's true speed, acc, direction at frame i, how well can we predict next frame assuming they are constant?
#' 
#' 

#add predictions to train
train_pred = train %>%
  #calculate position in next frame using true s, a, dir
  mutate(pred_dist_diff = est_speed*0.1 + est_acc_vector*0.5*0.1^2, #using speed + acc
         pred_dist_diff_2 = est_speed*0.1, #using speed only
         pred_x_sa = x + cos(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff,
         pred_y_sa = y + sin(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff,
         pred_x_s = x + cos(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff_2,
         pred_y_s = y + sin(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff_2, #make sure you convert the angle back to original coordinate system, since using trig functions
  ) %>%
  #shift the predictions to the leading row (were predicting the next frame)
  #this way were only using current/past data to predict future
  group_by(game_id, nfl_id, play_id) %>%
  mutate(across(starts_with("pred_"), lag)) %>%
  ungroup()

#' the predicted position from the next frame is the exact same movement from the previous frame
#' that is, the player is going the same speed (travels the same distance) in the same direction

#direction of previous frame vs direction of next frame
group_id = 1 #single player on single play
plot_df = train_pred %>% 
  filter(game_player_play_id == group_id)

#plot direction
ggplot(data = plot_df, aes(x = est_dir, y = lead(est_dir))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  scale_x_continuous(limits = c(0, 360), breaks = c(0, 90, 180, 270, 360)) +
  scale_y_continuous(limits = c(0, 360), breaks = c(0, 90, 180, 270, 360)) +
  labs(x = "Direction in Current Frame", y = "Direction in Next Frame") +
  theme_bw()

#clearly this assumption doesn't make sense, this would mean player is travelling in straight line the whole time
#same with speed and acceleration, they aren't the same throughout the play

ggplot(data = plot_df, aes(x = est_speed, y = lead(est_speed))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "Speed in Current Frame", y = "Speed in Next Frame") +
  theme_bw()

ggplot(data = plot_df, aes(x = est_acc_vector, y = lead(est_acc_vector))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  labs(x = "Acceleration in Current Frame", y = "Acceleration in Next Frame") +
  theme_bw()


#' lets see how well it can predict using the true x,y coordinates

#predicted position vs actual position
plot_df %>% 
  rename(pred_x = pred_x_s, pred_y = pred_y_s) %>%
  pivot_longer(cols = c(x, y, pred_x, pred_y),
               names_to = c("obs", ".value"),
               names_pattern = "(pred_)?(.*)") %>%
  mutate(obs = ifelse(obs == "", "Recorded", "Estimated")) %>%
  ggplot(mapping = aes(x = x, y = y, colour = obs)) + 
  geom_point() +
  #geom_text_repel(aes(label = frame_id)) +
  scale_x_continuous(n.breaks = 20) +
  scale_y_continuous(n.breaks = 20) +
  theme_bw()
  

#some minor differences between using s only vs s + a


#get predictions post-throw
test_preds = plot_df %>% filter(throw == "post") %>% select(frame_id, x, y, pred_x_sa, pred_y_sa, pred_x_s, pred_y_s)

#speed + acceleration
get_rmse(true_x = test_preds$x, true_y = test_preds$y, 
         pred_x = test_preds$pred_x_sa, pred_y = test_preds$pred_y_sa)
#speed only
get_rmse(true_x = test_preds$x, true_y = test_preds$y, 
         pred_x = test_preds$pred_x_s, pred_y = test_preds$pred_y_s)


#summarise rmse across all players
#compare if using acceleration is better or not
train_pred_summary = train_pred %>% 
  filter(throw == "post") %>%
  group_by(game_id, nfl_id, play_id) %>%
  summarise(rmse_s = get_rmse(true_x = x, true_y = y,
                              pred_x = pred_x_s, pred_y = pred_y_s),
            rmse_sa = get_rmse(true_x = x, true_y = y,
                               pred_x = pred_x_sa, pred_y = pred_y_sa))

#plot
train_pred_summary %>% 
  pivot_longer(cols = starts_with("rmse"), 
               names_to = "calc",
               values_to = "value") %>%
  ggplot(mapping = aes(x = calc, y = value)) +
  geom_boxplot() +
  theme_bw()


#overall using speed + acceleration gives tiny better rmse on average
train_pred_summary$rmse_sa %>% mean()
train_pred_summary$rmse_s %>% mean()

#rmse over entire dataset
train_pred %>% 
  filter(throw == "post") %>% 
  summarise(rmse_s = get_rmse(true_x = x, true_y = y,
                              pred_x = pred_x_s, pred_y = pred_y_s),
            rmse_sa = get_rmse(true_x = x, true_y = y,
                               pred_x = pred_x_sa, pred_y = pred_y_sa))

#' including speed + acceleration better


#' this confirms that getting good predictions is dependent on getting good speed, acceleration, and direction at each frame
#' but getting good speed, acceleration, and direction depends on good location
#' its circular
#' I think the way to go is to iterate by conditioning on eachother until convergance, like markov chain, gibbs





############################################### Baseline Model ############################################### 

#' this is a simple model that can then be built upon
#' autoregressive, predicts position in next frame from current frame
#' need three seperate models: direction, speed, acceleration
#' once you have these three things, then its a simple kinematics formula


#' ignore the player info and everything else right now, just keep it simple
#' 
#' also right now this is only using players_to_predict, not sure if it makes sense to include the other players?



#' first fit model on entire dataset
#' not worried about tuning right now

data_mod = train %>% 
  #order the columns
  select(est_dir, est_speed, est_acc_vector, game_player_play_id, everything()) %>%
  #remove NA responses
  filter(!is.na(est_dir) & !is.na(est_speed) & !is.na(est_acc_vector)) %>%
  #unselect unnecessary feature columns - things that can't be calculated post throw
  select(-c(game_id, nfl_id, play_id, s, a, o, dir, player_to_predict, player_birth_date,
            est_acc_scalar)) %>%
  rename(s = est_speed, a = est_acc_vector, dir = est_dir) %>%
  #add previous dir,s,a as features
  group_by(game_player_play_id) %>%
  mutate(prev_dir = lag(dir),
         prev_s = lag(s),
         prev_a = lag(a)) %>%
  ungroup()
  
#these have the true x,y values but that's ok
#fitting it on true value calculations is ok, but we just can't use the true values to calculate when we actually predict

#' 80% train, 20% test - by game_player_play groups
set.seed(1999)
n_game_player_plays = train %>% pull(game_player_play_id) %>% unique() %>% length() #46,045
split = sample(unique(data_mod$game_player_play_id), size = round(0.8*n_game_player_plays))
curr_train = data_mod %>% filter(game_player_play_id %in% split)
curr_test = data_mod %>% filter(!(game_player_play_id %in% split))

#fit models
xg_train_df = curr_train %>% select(-c(game_player_play_id, game_play_id, 
                                       x, y, frame_id, ball_land_x, ball_land_y, 
                                       num_frames_output, num_frames))

dir_xg = xgboost(data =  data.matrix(xg_train_df[,-1]), 
                 label = curr_train$dir,
                 nrounds = 200, print_every_n = 10)

speed_xg = xgboost(data =  data.matrix(xg_train_df[,-2]), 
                   label = curr_train$s,
                   nrounds = 200, print_every_n = 10)

acc_xg = xgboost(data =  data.matrix(xg_train_df[,-3]), 
                 label = curr_train$a,
                 nrounds = 200, print_every_n = 10)

#feature importance
xgb.importance(model = dir_xg)
xgb.importance(model = speed_xg)
xgb.importance(model = acc_xg)


#maybe model the change in direction, speed, acc...

#now use fits on test set to predict position on future frame, then recalculate kinematics, then predict future frame, recalculate, ... repeat

#just predict on post throw
curr_test = curr_test %>% 
  filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) #filter for post throw only

#what if you fit the models only on post throw?
#or maybe filter out the first few frames in each play

#for testing
#curr_test = curr_test[1:1000,]

#storing restuls
results_pred = list()

#loop
set.seed(1999)
for (i in 1:nrow(curr_test)) {
  if(i %% 10000 == 0) {print(paste0(round(i/nrow(curr_test), 2), " complete"))} #see progress
  
  curr_row = curr_test[i,]

  #initialize position as last observed values before throw
  if (curr_row$throw == "pre") { 
    #if last observation pre throw, predict next frame position using true observed kinematic values
    pred_dist_diff = curr_row$s*0.1 + curr_row$a*0.5*0.1^2
    
    pred_x = curr_row$x + cos(((90 - curr_row$dir) %% 360)*pi/180)*pred_dist_diff
    pred_y = curr_row$y + sin(((90 - curr_row$dir) %% 360)*pi/180)*pred_dist_diff
    
    #predict dir, s, a
    xg_pred_df = curr_row %>%  #row to predict on
      select(-c(game_player_play_id, game_play_id, x, y,
                frame_id, ball_land_x, ball_land_y,
                num_frames_output, num_frames)) 
    
    pred_dir = predict(dir_xg, data.matrix(xg_pred_df[,-1]))
    pred_s = predict(speed_xg, data.matrix(xg_pred_df[,-2]))
    pred_a = predict(acc_xg, data.matrix(xg_pred_df[,-3]))
    
  } else {
    prev_row = results_pred[[i-1]]
    
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
    
    x_diff = curr_row$x_diff = curr_x - prev_x
    y_diff = curr_row$y_diff = curr_y - prev_y
    
    #current dir, s, a
    curr_dir = prev_row$pred_dir = curr_row$dir 
    curr_s = prev_row$pred_s = curr_row$s
    curr_a = prev_row$pred_a = curr_row$a
    
    #other features
    curr_row$dir_diff = min(
      (curr_row$dir - prev_row$dir) %% 360,
      (-(curr_row$dir - prev_row$dir)) %% 360
    )
    curr_row$dist_diff = get_dist(x_diff = x_diff, y_diff = y_diff)
    curr_row$curr_ball_land_dir = get_dir(x_diff = curr_row$ball_land_x - curr_x, y_diff = curr_row$ball_land_y - curr_y)
    curr_row$dist_ball_land = get_dist(x_diff = curr_row$ball_land_x - curr_x, y_diff = curr_row$ball_land_y - curr_y)
    
    curr_row$ball_land_dir_diff = min(
      (curr_row$dir - curr_row$curr_ball_land_dir) %% 360,
      (-(curr_row$dir - curr_row$curr_ball_land_dir)) %% 360
    ) 
    
    curr_row$ball_land_diff_x = curr_row$ball_land_x - curr_x
    curr_row$ball_land_diff_y = curr_row$ball_land_y - curr_y
    
    
    #predict dir, s, a
    xg_pred_df = curr_row %>% #row to predict on
      select(-c(game_player_play_id, game_play_id, x, y,
                frame_id, ball_land_x, ball_land_y,
                num_frames_output, num_frames))
    
    pred_dir = predict(dir_xg, data.matrix(xg_pred_df[,-1]))
    pred_s = predict(speed_xg, data.matrix(xg_pred_df[,-2]))
    pred_a = predict(acc_xg, data.matrix(xg_pred_df[,-3]))
    
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
  results_pred[[i]] = curr_row
}
#def need to do this in parallel

#bind results into df
results_pred = results_pred %>%
  bind_rows() %>%
  #join true x,y values
  mutate(true_x = curr_test$x,
         true_y = curr_test$y)

group_id = 2

results_pred_single_play = results_pred %>% 
  group_by(game_player_play_id) %>%
  filter(cur_group_id() == group_id) %>%
  ungroup() %>%
  select(game_player_play_id, frame_id, x, true_x, y, true_y, dir, pred_dir, s, pred_s, a, pred_a) %>% 
  rename(pred_x = x, pred_y = y)
results_pred_single_play

#plot_player_movement(unique(results_pred_single_play$game_player_play_id))
#game_player_play_id 11 def a defense, running away from ball after it's thrown?

plot_player_movement_pred(group_id = unique(results_pred_single_play$game_player_play_id),
                          group_id_preds = results_pred_single_play %>% select(frame_id, pred_x, pred_y))

#now plot multiple players on same play with predictions
multi_player_pred_single_play = results_pred %>% 
  group_by(game_play_id) %>%
  filter(cur_group_id() == group_id) %>%
  ungroup() %>%
  select(game_play_id, game_player_play_id, frame_id, x, true_x, y, true_y, dir, pred_dir, s, pred_s, a, pred_a) %>% 
  rename(pred_x = x, pred_y = y)
multi_player_pred_single_play

multi_player_movement(group_id = unique(multi_player_pred_single_play$game_play_id))
multi_player_movement_pred(group_id = unique(multi_player_pred_single_play$game_play_id),
                           group_id_preds = multi_player_pred_single_play %>% select(game_player_play_id, frame_id, pred_x, pred_y))



# #predicted position vs actual position
# plot_df = results_pred %>%  
#   group_by(game_player_play_id) %>%
#   filter(cur_group_id() == group_id,
#          throw == "post") %>%
#   pivot_longer(cols = c(x, y, true_x, true_y),
#                names_to = c("obs", ".value"),
#                names_pattern = "(true_)?(.*)") %>%
#   mutate(obs = ifelse(obs == "", "Predicted", "True"))
# 
# #plot
# ggplot(plot_df, mapping = aes(x = x, y = y, colour = obs, label = frame_id)) + 
#   geom_point() +
#   #geom_text_repel() +
#   scale_x_continuous(n.breaks = 20) +
#   scale_y_continuous(n.breaks = 20) +
#   theme_bw()



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

#0.94





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



