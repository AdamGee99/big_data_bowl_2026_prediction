############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(data.table)
library(gridExtra)
library(patchwork)
library(scales)


############################################### Import Data ############################################### 

train_X_files = list.files(path = here("data", "train"), pattern = "input")
train_Y_files = list.files(path = here("data", "train"), pattern = "output")

train_X = train_X_files %>% map_dfr(~read.csv(here("data", "train", .x))) #like 5 million rows
train_Y = train_Y_files %>% map_dfr(~read.csv(here("data", "train", .x)))

# these are the pre-throw features
head(train_X)

#these are the post-throw player positions
head(train_Y)

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

#the features that are constant for a plyaer across a play
stable_features = train_X %>% 
  group_by(game_id, nfl_id, play_id) %>%
  select(player_to_predict, play_direction, absolute_yardline_number, player_name, player_height, 
         player_weight, player_birth_date, player_position, player_side, player_role, ball_land_x, ball_land_y) %>%
  slice(1)

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

#' iteratively add speed, acceleration, direction, orientation based on observed x,y coordinates

#' speed is in yards/s, acceleration is in yards/s^2, 
#' direction is angle of player motion in degrees, orientation is orientation of player in degrees (these might be hard to calculate)


#' make function that plots recorded vs estimated speed and acceleration from x,y

#group_id is a singl player on a single play
#lag_frames is the number of previous frames to calculate s, a
#lead is the number of future frames to calculate s, a
plot_speed_acc = function(group_id, lag_frames, lead_frames) {
  window_size = lag_frames + lead_frames #num frames to calculate s, a over
  
  #single player on a single play
  plot_df = train %>%
    filter(player_to_predict) %>%
    group_by(game_id, nfl_id, play_id) %>%
    filter(cur_group_id() == group_id) %>%
    select(game_id, play_id, throw, s, a, dir, o, frame_id, player_name, x, y) %>%
    ungroup() %>%
    #calculations below
    mutate(x_diff = lead(x, n = lead_frames) - lag(x, n = lag_frames), #the difference in x direction from previous frame in yards
           y_diff = lead(y, n = lead_frames) - lag(y, n = lag_frames), #the difference in y direction from previous frame in yards
           distance_diff = sqrt(x_diff^2 + y_diff^2), #distance travelled from previous frame in yards
           est_speed = distance_diff/((window_size)/10), #yards/second (1 frame is 0.1 seconds)
           est_acc_vector = (lead(est_speed, n = lead_frames) - lag(est_speed, n = lag_frames))/((window_size)/10), #this has directions (negative accelerations)
           est_acc_scalar = abs(est_acc_vector)) %>% #this is magnitude of acceleration (only positive) 
    #pivot longer for plotting
    pivot_longer(cols = c(s, a, est_speed, est_acc_vector, est_acc_scalar),
                 names_to = c("obs", ".value"),
                 names_pattern = "(est_)?(.*)") %>%
    mutate(obs = ifelse(obs == "", "Recorded", "Estimated"),
           s = ifelse(is.na(s), speed, s),
           a = ifelse(is.na(a), acc_scalar, a),
           a_vec = ifelse(is.na(a), acc_vector, a)) %>%
    select(-c(speed, acc_scalar, acc_vector))
  
  #make speed plot
  speed_plot = ggplot(data = plot_df, mapping = aes(x = frame_id, y = s, colour = obs)) +
    geom_point() +
    theme_bw() +
    labs(x = "", y = "Speed (yds/sec)", title = "Speed") +
    theme(legend.title = element_blank(),
          legend.position = "bottom")
  
  #make acceleration plot
  acc_plot =  ggplot(data = plot_df, mapping = aes(x = frame_id, y = a, colour = obs)) +
    geom_point() +
    theme_bw() +
    labs(x = "Frame ID", y = "Acceleration (yds/sec^2)", title = "Acceleration") +
    theme(legend.title = element_blank(),
          legend.position = "bottom")
  
  #plot both
  wrap_plots(list(speed_plot, acc_plot),
             nrow = 2) +
    plot_layout(guides = "collect") & theme(legend.position = "bottom")
}

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

plot_dir = function(group_id, lag_frames, lead_frames) {
  plot_df = train %>%
    filter(player_to_predict) %>%
    group_by(game_id, nfl_id, play_id) %>%
    filter(cur_group_id() == group_id) %>%
    select(game_id, play_id, throw, s, a, dir, o, frame_id, player_name, x, y) %>%
    ungroup() %>%
    #calculations below
    mutate(x_diff = lead(x, n = lead_frames) - lag(x, n = lag_frames), #the difference in x direction from previous frame in yards
           y_diff = lead(y, n = lead_frames) - lag(y, n = lag_frames), #the difference in y direction from previous frame in yards
           est_dir = (90 - (atan2(y = y_diff, x = x_diff)*180/pi)) %% 360) %>% 
    pivot_longer(cols = c("dir", "est_dir"), names_to = "obs", values_to = "dir") %>%
    mutate(obs = ifelse(obs == "dir", "Recorded", "Estimated"))
  
  #plot
  ggplot(data = plot_df, mapping = aes(x = frame_id, y = dir, colour = obs, shape = obs)) +
    geom_point(aes(size = obs)) +
    scale_shape_manual(values = c("Estimated" = 19, "Recorded" = 8)) +
    scale_size_manual(values = c("Estimated" = 3, "Recorded" = 6)) +
    scale_y_continuous(limits = c(0, 360), breaks = c(0, 90, 180, 270, 360)) +
    labs(x = "Frame ID", y = "Player Direction (deg)") +
    theme_bw() +
    theme(legend.title = element_blank(),
          legend.position = "bottom")
}

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
game_ids = train$game_id %>% unique() #272 games
nfl_ids = train$nfl_id %>% unique() #1384 player ids

#visualise single player's movement

#function only plots players that were targeted
plot_player_movement = function(group_id) {
  plot_df = train %>% 
    filter(player_to_predict) %>% #filter for only players that were targeted
    group_by(game_id, nfl_id, play_id) %>% 
    filter(cur_group_id() == group_id) %>% #filter for a single player on a single play
    select(frame_id, x, y, ball_land_x, ball_land_y, throw, player_side) %>%
    ungroup() 
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  nfl_id = plot_df$nfl_id %>% unique()
  
  #plot
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = frame_id, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_gradient(low = "black", high = "green") +
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y), colour = "red", size = 4) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id, ", Player ID: ", nfl_id)) +
    theme_bw() +
    guides(colour = "none", shape = "none")
}

plot_player_movement(group_id = 1)
plot_player_movement(group_id = 2)

# #plotting a grid 
# plots_1 = lapply(1:9, plot_player_movement)
# plots_2 = lapply(10:18, plot_player_movement)
# plots_3 = lapply(19:27, plot_player_movement)
# 
# grid.arrange(grobs = plots_1, ncol = 3, nrow = 3)
# grid.arrange(grobs = plots_2, ncol = 3, nrow = 3)
# grid.arrange(grobs = plots_3, ncol = 3, nrow = 3)
# 
# num_plots = 16
# grid.arrange(grobs = lapply(1:num_plots, plot_player_movement),
#              ncol = ceiling(sqrt(num_plots)),
#              nrow = ceiling(sqrt(num_plots)))


#function now plotting multiple players on same play
multi_player_movement = function(group_id) {
  plot_df = train %>% 
    filter(player_to_predict) %>% #filter for only players that were targeted
    group_by(game_id, play_id) %>% 
    filter(cur_group_id() == group_id) %>% #filter for a single play
    select(game_id, play_id, frame_id, x, y, ball_land_x, ball_land_y, throw, player_side, player_name) %>%
    mutate(colour = ifelse(
      player_side == "Offense", 
      col_numeric(c("black", "green"), domain = range(frame_id))(frame_id),
      col_numeric(c("black", "blue"), domain = range(frame_id))(frame_id)
    )) %>% #add colours depending on offense or defense
    ungroup() 
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  
  #plot 
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = colour, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_identity() + 
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y), colour = "red", size = 4) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id)) +
    theme_bw() +
    guides(shape = "none")
}
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
group_id = 1
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
#' Once you have that, predicting the next frame seems very simple (like you don't even need a model, you can predict the value exactly using physics formula...)
#' Use all the extra features (where ball is landing, player info, defender info, ...) to predict direction/speed/acc






############################################### TO DO ############################################### 

#' Steps
#' 1. get model to predict player's next frame position based on direction, speed, acc of current frame
#'      -this can be an exact formula, no estimation...
#'      -see how well this matches up with true observed values
#'      -figure out what features are needed here (is direction, speed, acceleration all or is more neeeded? orientation?)
#' 
#' 2. start developing models for the each features needed above (direction, speed, acceleration)





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



