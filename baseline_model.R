############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(ggrepel)

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

#just use previous frame only for calculations for now
lead_frames = 0
lag_frames = 1
window_size = lag_frames + lead_frames

#add predictions to train
train_pred = train %>%
  filter(player_to_predict) %>%
  group_by(game_id, nfl_id, play_id) %>%
  #filter(cur_group_id() == group_id) %>% #use this to filter for only 1 player in 1 play in 1 game
  select(game_id, play_id, throw, s, a, dir, o, frame_id, player_name, x, y) %>%
  #first calculate s, a, dir from true values of x,y after throw
  mutate(x_diff = lead(x, n = lead_frames) - lag(x, n = lag_frames, default = NA), #the difference in x direction from previous frame in yards
         y_diff = lead(y, n = lead_frames) - lag(y, n = lag_frames, default = NA), #the difference in y direction from previous frame in yards
         dist_diff = sqrt(x_diff^2 + y_diff^2), #distance travelled from previous frame in yards
         est_speed = dist_diff/((window_size)/10), #yards/second (1 frame is 0.1 seconds)
         est_acc_vector = (lead(est_speed, n = lead_frames) - lag(est_speed, n = lag_frames))/((window_size)/10), #this has directions (negative accelerations)
         est_acc_scalar = abs(est_acc_vector),
         est_dir = get_dir(x_diff = x_diff, y_diff = y_diff)) %>% 
  ungroup() %>%
  #now calculate position in next frame using true s, a, dir
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
  group_by(game_id, nfl_id, play_id) %>% 
  filter(cur_group_id() == group_id) %>%
  ungroup() %>%
  select(-c(game_id, nfl_id, play_id, o, player_name))

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


#' lets see how well it can predict using the true x,y coordinates though

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
#' but getting good speed, acceleration, adn direction depends on good location
#' its circular
#' I think the way to go is to iterate by conditioning on eachother until convergance, like markov chain, gibbs





#' now lets try using previous prediction as the previous position, rather than true position values
#' pretty sure this will just result in a straight line but lets try it
#' 

test = train %>%
  group_by(game_id, nfl_id, play_id) %>%
  filter(cur_group_id() == group_id) %>%
  ungroup()

test = test[-c(1:10),]

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



#' next step is to create a DAG










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



