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



#' now lets try using previous prediction as the previous position, rather than true position values
#' pretty sure this will just result in a straight line but lets try it

test = plot_df[-c(1:10),]

#preds = data.frame("x" = NULL, "y" = NULL, "dir" = NULL, "s" = NULL, "a" = NULL)
preds = matrix(ncol = 5, nrow = nrow(test),
               dimnames = list(seq(1:nrow(test)),
                               c("x", "y", "dir", "s", "a")))
for (i in 1:nrow(test)) {
  curr_frame = test[i,] #the current frame and all the info
  
  if (i %in% c(1,2)) {
    preds_x[i] = curr_frame$x
    preds_y[i] = curr_frame$y
    
    #initialize first two rows by true observations
    preds[i,] = c(curr_frame$x, curr_frame$y, 
                  curr_frame$dir, curr_frame$s, curr_frame$a)
  }
  else {
    prev_x = preds[i-2, 1]
    prev_y = preds[i-2, 2]
    
    curr_x = preds[i-1, 1]
    curr_y = preds[i-1, 2]
    
    x_diff = curr_x - prev_x
    y_diff = curr_y - prev_y
    dist_diff = get_dist(x_diff, y_diff)

    est_dir = get_dir(x_diff, y_diff) #direction change between two frames
    est_speed = dist_diff/0.1 #just using 1 frame for now
    est_acc = (preds[i-1, 4] - preds[i-2, 4])/0.1 #change in speed over 1 frame
    
    #update position
    pred_dist_diff = est_speed*0.1 + est_acc*0.5*0.1^2 #predicted distance travelled between frames - using speed + acceleration
    pred_x = prev_x + cos(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff
    pred_y = prev_y + sin(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff
    
    #store
    preds[i,] = c(pred_x, pred_y, est_dir, est_speed, est_acc)
  }
} 
#preds = preds %>% as.data.frame()
#colnames(preds) = c("x", "y", "dir", "s", "a")
preds


#' can eventually turn this into a function where you supply the pre throw info and it predicts all the post throw frames

#' create a DAG





############################################### Baseline Model ############################################### 

#' this is a simple model that can then be built upon
#' autoregressive, predicts position in next frame from current frame
#' need three seperate models: direction, speed, acceleration
#' once you have these three things, then its a simple kinematics formula


#' ignore the player info and everything else right now, just keep it simple
#' 
#' also right now this is only using players_to_predict, not sure if it makes sense to include the other players?



#' use cross-validation by splitting up different game-player-plays groups
num_folds = 10
n_game_player_plays = train %>% pull(game_player_play_id) %>% unique() %>% length() #46,045

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
  
  #store predictions
  results[[fold]] = curr_test %>%
    mutate(dir_pred = predict(curr_dir_xg, data.matrix(curr_test[,-c(1, 4)])),
           speed_pred = predict(curr_speed_xg, data.matrix(curr_test[,-c(2, 4)])),
           acc_pred = predict(curr_acc_xg, data.matrix(curr_test[,-c(3, 4)])))
}

#now use predicted dir, speed, acc to predict positions
results_pred = results %>% 
  bind_rows() %>%
  #take the mean if plays were in multiple plays
  group_by(game_player_play_id, frame_id) %>%
  mutate(dir_pred = mean(dir_pred),
         speed_pred = mean(speed_pred),
         acc_pred = mean(acc_pred)) %>%
  ungroup() %>%
  filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) %>% #filter for post throw only
  #update position
  mutate(pred_dist_diff = speed_pred*0.1 + acc_pred*0.5*0.1^2)


preds = matrix(ncol = 2, 
               nrow = nrow(results_pred),
               dimnames = list(seq(1:nrow(results_pred)), c("x", "y")))
#loop
for (i in 2:nrow(results_pred)) {
  if(i %% 10000 == 0) {print(paste0(round(i/nrow(results_pred), 2), " complete"))} #see progress
  
  prev_row = results_pred[i-1,]
  curr_row = results_pred[i,]
  
  #initialize at last observed values before throw
  if (prev_row$throw == "pre") {
    prev_x = prev_row$x
    prev_y = prev_row$y
  } else {
    #previous x,y from past prediction
    prev_x = preds[i-1,1]
    prev_y = preds[i-1,2]
  }
  #new prediction
  pred_x = prev_x + cos(((90 - curr_row$dir_pred) %% 360)*pi/180)*curr_row$pred_dist_diff
  pred_y = prev_y + sin(((90 - curr_row$dir_pred) %% 360)*pi/180)*curr_row$pred_dist_diff
    
  #store
  preds[i,1] = pred_x
  preds[i,2] = pred_y
}

#add predictions back to results df
results_pred$pred_x = preds[,1]
results_pred$pred_y = preds[,2]


#predicted position vs actual position
plot_df = results_pred %>%  
  group_by(game_player_play_id) %>%
  filter(cur_group_id() == 4,
         throw == "post") %>%
  pivot_longer(cols = c(x, y, pred_x, pred_y),
               names_to = c("obs", ".value"),
               names_pattern = "(pred_)?(.*)") %>%
  mutate(obs = ifelse(obs == "", "True Position", "Predicted Position"))

#plot
ggplot(plot_df, mapping = aes(x = x, y = y, colour = obs, label = frame_id)) + 
  geom_point() +
  #geom_text_repel(data = plot_df %>%filter(obs == "True Position")) +
  scale_x_continuous(n.breaks = 20) +
  scale_y_continuous(n.breaks = 20) +
  theme_bw()



#get rmse





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



