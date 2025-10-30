############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(data.table)
library(gridExtra)
library(patchwork)
library(scales)
library(scattermore)


############################################### Import Data ############################################### 

train_X_files = list.files(path = here("data", "train"), pattern = "input")
train_Y_files = list.files(path = here("data", "train"), pattern = "output")

train_X = train_X_files %>% map_dfr(~read.csv(here("data", "train", .x))) #like 5 million rows
train_Y = train_Y_files %>% map_dfr(~read.csv(here("data", "train", .x)))

# these are the pre-throw features
head(train_X)

#these are the post-throw player positions
head(train_Y)

#source helper functions
source(here("helper.R"))


############################################### Cleaning ############################################### 


#clean the types
str(train_X)
train_X = train_X %>% mutate(player_to_predict = as.logical(player_to_predict), #change player_to_predict to logical
                             player_birth_date = ymd(player_birth_date), #change player_birth_date to date
                             across(where(is.character), as.factor)) #change remaining character types to factors

str(train_Y) #nothing to change



#join datasets (add movement after the throw to train_X)

#adding column to train_X and train_Y indicating pre or post throw
train_X = train_X %>% mutate(throw = "pre")
train_Y = train_Y %>% mutate(throw = "post")

#the features that are constant for a player across a play
stable_features = train_X %>% 
  group_by(game_id, nfl_id, play_id) %>%
  select(player_to_predict, play_direction, absolute_yardline_number, player_name, player_height, 
         player_weight, player_birth_date, player_position, player_side, player_role, ball_land_x, ball_land_y) %>%
  dplyr::slice(1) 

train = train_X %>% 
  filter(player_to_predict) %>% #filter for only players that were targeted
  group_by(game_id, nfl_id, play_id, num_frames_output) %>% 
  summarise(pre = n()) %>% #the number of frames pre throw
  rename(post = num_frames_output) %>% #the number of frames post throw
  pivot_longer(cols = c(pre, post), names_to = "throw", values_to = "num_frames") %>%
  #now join train_X on pre rows
  full_join(train_X, by = c("game_id", "nfl_id", "play_id", "throw")) %>%
  #now merge train_Y on post rows
  full_join(train_Y, by = c("game_id", "nfl_id", "play_id", "throw")) %>%
  #join the stable features
  left_join(stable_features, by = c("game_id", "nfl_id", "play_id")) %>%
  ungroup() %>%
  pivot_longer(
    cols = c(ends_with(".x"), ends_with(".y")),      # the columns created by full_join
    names_to = c(".value", "source"),                # splits into "x"/"y" and their source ("x"/"y")
    names_sep = "\\."
  ) %>%
  select(-source) %>%                  # remove source column
  drop_na(x, y, frame_id) %>% 
  #mutate frame id to be continuous
  group_by(game_id, nfl_id, play_id) %>%
  mutate(frame_id = row_number()) %>%
  ungroup()
  
head(train)
rm(train_X, train_Y)
#' speed is in yards/s, acceleration is in yards/s^2, 
#' direction is angle of player motion in degrees, orientation is orientation of player in degrees (these might be hard to calculate)

#save cleaned data
#write.csv(train, file = here("data", "train_clean.csv"), row.names = FALSE)


#group_id is a single player on a single play
#lag_frames is the number of previous frames to calculate s, a
#lead is the number of future frames to calculate s, a
plot_speed_acc(group_id = 1, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_speed_acc(group_id = 1, lag_frames = 1, lead_frames = 0) #using no future data

plot_speed_acc(group_id = 22, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_speed_acc(group_id = 22, lag_frames = 1, lead_frames = 0) #using no future data

plot_speed_acc(group_id = 432, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_speed_acc(group_id = 432, lag_frames = 1, lead_frames = 0) #using no future data

#' using lead frame seems to give better estimates of speed and acceleration

#' 1 frame of lag and 1 frame of lead seems to give best estimate compared to recorded, good balance of smooth over two frames but not too smooth


#' I think what we need to do here is use a model that predicts frame_id + 1 x and y while only using previous frame's x,y, 
#' then use that prediction to get better estimates of current speed/acceleration, 
#' then use another model to get final predictions using the better curr speed/acc - maybe keep iterating until predictions converge


#' can I use the fact the speed and acceleration right before throw was calculated using x,y's after throw to my advantage????

#' I swear some recorded accelerations are wrong (cur_group_id == 1, cur_group_id == 2823), acceleration becomes way too high right before throw
#' I think maybe drop recorded a and just use your estimated a from the player's positioning
#' but experiment with this, see what gives better CV




#' now estimate orientation and direction of motion!!
#' 
#' 
#' I think direction is simple to calculate but orientation might be unknown
#' what if we make a model to predict a players orientation then use that in subsequent final model?

#not sure if we need to use leading frames here..
plot_dir(group_id = 1, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_dir(group_id = 1, lag_frames = 1, lead_frames = 0) #using no future data

plot_dir(group_id = 22, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_dir(group_id = 22, lag_frames = 1, lead_frames = 0) #using no future data

plot_dir(group_id = 432, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_dir(group_id = 432, lag_frames = 1, lead_frames = 0) #using no future data


#again we see that using lead frame is important here, 
#do the same process as above? predict from past x,y ... update past x,y, predict from better past x,y ... repeat until convergence...



#direction should be very important, like especially at speed, the player is going to continue going in the same direction they are going



## orientation should be important but it's impossible to estimate purely from x,y positions. They use sensors in players shoulder pads...

#I guess skip using it for now, but come back to it later






############################################### EDA ############################################### 


#first lets visualize the player's movements
train$game_id %>% unique() %>% length() #272 games
train$nfl_id %>% unique() %>% length() #1384 player ids

plot_player_movement(group_id = 1)
plot_player_movement(group_id = 2)

multi_player_movement(1)

num_plots = 4
wrap_plots(lapply(1:num_plots, multi_player_movement),
           ncol = ceiling(sqrt(num_plots)),
           nrow = ceiling(sqrt(num_plots))) +
  plot_layout(guides = "collect")
#blue is defense, green is offense

#filtering for all players in play gives interesting results too 




### see the relationships between response and features now

#find out whats important

#plot player movement with direction
group_id = 53
wrap_plots(list(plot_player_movement(group_id = group_id),
                plot_dir(group_id = group_id, lag_frames = 1, lead_frames = 1)),
           nrow = 2) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
#clearly very important, tells us where the player is heading
#probably the most important feature


#plot player movement with speed, acceleration
wrap_plots(list(plot_player_movement(group_id = group_id),
                plot_speed_acc(group_id = group_id, lag_frames = 1, lead_frames = 1)),
           nrow = 2) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
#also important, tells us how far away the position at the next frame will be


#' I think the challenge is getting the correct direction, speed, acc of player in the current frame
#' at every frame, we should be predicting not just the next x,y but also need to update dir, s, a 
#' these will also have to be models, but they all depend on eachtother so it's kind of weird...
#' Once you have that, predicting the next frame seems very simple (like you don't even need a model, you can predict the value exactly using physics formula...)
#' Use all the extra features (where ball is landing, player info, defender info, ...) to predict direction/speed/acc




############################################### Deriving New Features ############################################### 

#' new features to add:
#'  -change in direction between current and previous frame
#'  -difference in current frames direction and direction of ball land (x,y)
#'  -distance where the player currently is to where the ball will land
#'  -proportion of play complete, standardizes frame ID to be all on the same scale

#' first add game_player_play_id and game_play_id
train = train %>% 
  group_by(game_id, play_id) %>%
  mutate(game_play_id = cur_group_id()) %>% #game_play_id
  ungroup() %>%
  group_by(game_id, nfl_id, play_id) %>%
  mutate(game_player_play_id = cur_group_id(),
         prop_play_complete = frame_id/max(frame_id)) %>% #proportion of play complete - standardizes frame ID %>%
  ungroup()

train_derived = train %>% 
  filter(player_to_predict) %>% #only derive the new features on player_to_predict
  est_kinematics() %>% #add estimated direction, speed, acceleration
  change_in_kinematics() %>% #add change in previous -> current and current -> next frame in direction, speed, acceleration
  derived_features() #add derived features

colnames(train_derived)

#save cleaned data
write.csv(train_derived, file = here("data", "train_clean.csv"), row.names = FALSE)





############################################### Continue EDA ############################################### 


#ball_land_dir_diff vs fut_dir_diff
ball_land_dir_diff_v_fut_dir_diff = ggplot(data = train, mapping = aes(x = ball_land_dir_diff, y = fut_dir_diff)) + 
  geom_scattermore(alpha = 0.05)
ball_land_dir_diff_v_fut_dir_diff

ball_land_dir_diff_v_fut_dir_diff + ylim(c(-30, 30))


#' negative relationship here makes sense
#' if the min distance to where you need to go is -15deg, then you should start heading positive
#' 
#' but whats up with these horizontal lines?



#' EDA on dir, speed, acc
#' try to visualize and understand the relationships between these and other predictors


#' speed vs pre-post throw
train_derived %>% 
  ggplot(mapping = aes(x = factor(throw), y = est_speed)) +
  geom_boxplot() + theme_bw()
#speed much higher after throw

#' acc vs pre-post throw
train_derived %>% 
  ggplot(mapping = aes(x = factor(throw), y = est_acc)) +
  geom_boxplot() + theme_bw() + ylim(c(-10, 10))
#acceleration more mixed, slightly higher pre throw actually
#makes sense, the player is usually running full speed after ball is in the air

#' ball_land_dir_diff vs pre-post throw
train_derived %>%
  ggplot(mapping = aes(x = factor(throw), y = abs(ball_land_dir_diff))) +
  geom_boxplot() + theme_bw()
#difference in direction to ball land smaller post throw, makes sense


#' speed vs direction change
train_derived %>%
  ggplot(mapping = aes(x = prev_dir_diff, y = est_speed)) +
  geom_scattermore() +
  theme_bw() 
#scale_x_continuous(limits = c(0, 50)) +
#scale_y_continuous(limits = c(0, 15))

#' when players are changing direction, they are going slower
#' direction is minimum of positve or negative direction (so 180 is maximum possible dir_diff)
#' 
#' some speed values here are clearly outliers potentially impossible - investigate these



#' acceleration vs direction change
train_derived %>%
  ggplot(mapping = aes(x = prev_dir_diff, y = est_acc)) +
  geom_scattermore() +
  scale_y_continuous(limits = c(-25, 25)) +
  theme_bw() 
#' same as above, when direction changing, acceleration is going to 0
#' this is kind of counter-intuitive
#' also some outliers here, look into those, I'm guessing its the values right at the start



#' speed vs acceleration 
#' 
#' this should be a simple relationship
#' just use all the observations for a single player, but across all games and plays

#acc scalar
train_derived %>% 
  #group_by(nfl_id) %>%
  #filter(cur_group_id() == 3) %>%
  ggplot(mapping = aes(x = cut(est_speed, breaks = c(0, 2, 4, 6, 8, 10, 30)), y = est_acc_scalar)) +
  #geom_scattermore() +
  geom_boxplot() +
  #scale_x_continuous(limits = c(0, 15)) +
  scale_y_continuous(limits = c(0, 10)) +
  labs(x = "Speed", y = "Acceleration") +
  #scale_colour_gradient(low = "black", high = "green") +
  theme_bw()
#there's some relationship here

#' ball_land_dir_diff vs frame_id
plot_ball_land_dir_diff(2)


#' generally, as play progresses the player's direction diff to ball land x,y gets closer to 0
#' as the player gets super close to the ball though things get weird since small changes in distances give big changes in angles, but doesn't matter much
#' therefore this feature is more important at farther distances, less important at closer distances
#' interaction with current distance to ball!!

#' at the end of the frames, when the ball is close to landing or being caught, the only thing that seems to matter is the kinematics part of the equation
#' ie, once player has narrowed in on where ball is going, only thing that matters is how fast they get there

#' also seems to be much more important for the offense, offense always wants to go to the ball, some defenders can drift around a bit



#' rather than making the goal to get to the exact point the ball will land, make the goal to get in some radius around this point
#' in reality, players just need to get close enough to catch it, not necessarily on the exact landing point
#' this might help clear up the issue of ball_land_dir_diff raising when player gets super close to landing point
#' I guess this is only for the offensive team

#' first do some EDA to see how close player needs to get to catch the ball, or how close they usually get
#' this could also be a tuning parameter? can adjust how big the radius should be to the landing point?

#' come back to this later





#' ball_land_dir_diff vs speed
train_derived %>% 
  # group_by(game_id, nfl_id, play_id) %>% 
  # filter(cur_group_id() == 1) %>%
  # ungroup() %>%
  ggplot(mapping = aes(x = cut(ball_land_dir_diff, breaks = c(0, 45, 90, 135, 180)), 
                       y = est_speed)) +
  geom_boxplot() +
  labs(x = "Ball Land Direction Difference", y = "Speed") + 
  #scale_y_continuous(limits = c(0, 180), breaks = c(0, 90, 180)) +
  #labs(x = "Frame ID", y = "Difference in current direction and ball land direction (deg)") +
  theme_bw()  

#higher speed when travelling where the ball will land
#makes sense





#' distance from ball vs speed ...
train_derived %>%
  ggplot(mapping = aes(x = cut(dist_ball_land, breaks = c(0, 10, 20, 30, 40, 50, 60)), y = est_speed)) +
  geom_boxplot() +
  labs(x = "Distance to ball landing location", y = "speed") +
  #geom_scattermore(alpha = 0.05) +
  scale_y_continuous(limits = c(0, 15)) +
  theme_bw()
#small relationship here


#' distance from ball vs acc ...
train_derived %>%
  ggplot(mapping = aes(x = cut(dist_ball_land, breaks = c(0, 10, 20, 30, 40, 50, 60)), y = est_acc_scalar)) +
  geom_boxplot() +
  labs(x = "Distance to ball landing location", y = "Acceleration") +
  #geom_scattermore(alpha = 0.05) +
  scale_y_continuous(limits = c(0, 20)) +
  theme_bw()
#acceleration higher when farther from ball, makes sense


#' distance from ball vs ball_land_diff_dir ...offense
train_derived %>%
  filter(player_side == "Offense") %>%
  ggplot(mapping = aes(x = cut(dist_ball_land, breaks = c(0, 10, 20, 30, 40, 50, 60)), y = ball_land_dir_diff)) +
  geom_boxplot() +
  labs(x = "Distance to ball landing location", y = "Difference to Ball Landing Direction") +
  theme_bw()

#' distance from ball vs ball_land_diff_dir defense
train_derived %>%
  filter(player_side == "Defense") %>%
  ggplot(mapping = aes(x = cut(dist_ball_land, breaks = c(0, 10, 20, 30, 40, 50, 60)), y = ball_land_dir_diff)) +
  labs(x = "Distance to ball landing location", y = "Difference to Ball Landing Direction") +
  geom_boxplot() +
  theme_bw()




#' none of these are linear relationships, distributions are weird and so many interactions... 
#' non-parametric def the way to go



############################################### Kinematics Part ############################################### 


#' use true s, a, dir to predict next frame
#' if you know the player's true speed, acc, direction at frame i, how well can we predict next frame assuming they are constant?
#' trying to show that if you know the true dir,s,a, then predicting the next x,y is trivial
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
#' 
#' 
#' 3. start thinking about how you will actually model
#'      -have to fit a model on each frame? seems way to expensive
#'      -separate model for s, dir, a, ...
#'      
#'      -start with the simplest things you can do as a baseline, then go from there
#'      
#'      
#' 4. update ball_land_dir_diff to be a radius around the ball
#'      -rather than making the goal to get to the exact point the ball will land, make the goal to get in some radius around this point
#'      -in reality, players just need to get close enough to catch it, not necessarily on the exact landing point
#'      -this might help clear up the issue of ball_land_dir_diff raising when player gets super close to landing point
#'      -I guess this is only for the offensive team
#'      
#'      -first do some EDA to see how close player needs to get to catch the ball, or how close they usually get
#'      -this could also be a tuning parameter? can adjust how big the radius should be to the landing point?





############################################### Misc Ideas I'll get to eventually ############################################### 

#' 1. check if theres anything in test thats not in train... (any player for eg)
#'      there will be

#' 2. add speed, direction, acceleration from predictions of model at every frame 
#'      recorded features s and a are magnitudes, not vectors
#'      experiment with including vectors (negative acceleration) in model, I think this makes a lot of sense - see which recorded or estimated gives better CV

#' 3. figure out how to incorporate orientation, cannot estimate from x,y - they use sensors in player's shoulder pads
#'      maybe estimate it? but how helpful will this even be?

#' 4. make a model to predict direction - then use that
#'      train model to predict direction, 
#'      if a player is turning for eg, the direction isn't just a straight line between current and previous frame, its going to keep curving
#'      this depends on speed, acc, and where the ball is landing
#'      two solutions to this, the gibbs way as before, or fit a model to predict direction

#' 5. certain players are dominant to one side, eg always like to cut right, use this info?



