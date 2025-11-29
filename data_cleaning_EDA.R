############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(data.table)
library(gridExtra)
library(patchwork)
library(scales)
library(scattermore)
source(here("helper.R"))


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
#' 
#' 
#' add game_player_play_id and game_play_id
#' these are unique ids for all the players in single plays and the plays
train = train %>% 
  group_by(game_id, play_id) %>%
  mutate(game_play_id = cur_group_id(), #game_play_id
         prop_play_complete = frame_id/max(frame_id)) %>% #proportion of play complete - standardizes frame ID
  ungroup() %>%
  group_by(game_id, nfl_id, play_id) %>%
  mutate(game_player_play_id = cur_group_id()) %>%
  ungroup() %>%
  #convert height to inches
  mutate(player_height = 12*as.numeric(sub("-.*", "", player_height)) + as.numeric(sub(".*-", "", player_height)))

train$game_play_id %>% unique() %>% length() #14,108 plays
train$game_player_play_id %>% unique() %>% length() #173,150 player-plays


############################################### Visualizing Kinematics ############################################### 

#group_id is a single player on a single play
plot_speed_acc(group_id = 1, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_speed_acc(group_id = 1, lag_frames = 1, lead_frames = 0) #using no future data

plot_speed_acc(group_id = 22, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_speed_acc(group_id = 22, lag_frames = 1, lead_frames = 0) #using no future data

plot_speed_acc(group_id = 432, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_speed_acc(group_id = 432, lag_frames = 1, lead_frames = 0) #using no future data
#' using lead frame seems to give better estimates of speed and acceleration
#' 1 frame of lag and 1 frame of lead seems to give best estimate compared to recorded, good balance of smooth over two frames but not too smooth

#' I think direction is simple to calculate but orientation might be unknown

#not sure if we need to use leading frames here..
plot_dir(group_id = 1, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_dir(group_id = 1, lag_frames = 1, lead_frames = 0) #using no future data

plot_dir(group_id = 22, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_dir(group_id = 22, lag_frames = 1, lead_frames = 0) #using no future data

plot_dir(group_id = 432, lag_frames = 1, lead_frames = 1) #using lead frame in calc
plot_dir(group_id = 432, lag_frames = 1, lead_frames = 0) #using no future data

#again we see that using lead frame is important here, 
#do the same process as above? predict from past x,y ... update past x,y, predict from better past x,y ... repeat until convergence...



############################################### Visualizing Player Movement ############################################### 

#3 players on the same play
plot_player_movement(group_id = 1)
plot_player_movement(group_id = 2)
plot_player_movement(group_id = 3)
#all of them on the same play
multi_player_movement(1)
#all players on the same play but include players_to_predict == FALSE
multi_player_movement(1, only_player_to_predict = FALSE)

#multiple plots at the same time
num_plots = 4
wrap_plots(lapply(1:num_plots, multi_player_movement),
           ncol = ceiling(sqrt(num_plots)),
           nrow = ceiling(sqrt(num_plots))) +
  plot_layout(guides = "collect")
#blue is defense, green is offense



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


### investigate any weird plays - some are super long and might be mis-recorded 
num_frames = train %>% filter(throw == "post") %>% group_by(game_player_play_id) %>% summarise(n_frames = n())
num_frames$n_frames %>% hist(breaks = 50)
num_frames$n_frames %>% summary()
num_frames$n_frames %>% quantile(probs = c(0.999))
#identify plays that run too long
long_player_plays = num_frames %>% filter(n_frames >= 50) %>% pull(game_player_play_id)
train %>% filter(game_player_play_id %in% c(long_player_plays)) %>%
  pull(game_play_id) %>% unique()
multi_player_movement_game_play_id(812) #this is clearly recorded wrong
multi_player_movement_game_play_id(11677) #this is fine

#get rid of this play
train = train %>% filter(!(game_play_id %in% 812))





############################################### Deriving New Features ############################################### 

#' new features to add:
#' 
#'   DONE:
#'  -change in dir, s, a between current and previous frame
#'  -difference in current frames direction and direction of ball land (x,y)
#'  -distance where the player currently is to where the ball will land
#'  -distance to nearest out of bounds point
#'  -direction to nearest out of bounds point
#'  
#'  -convert height to inches
#'  
#'  
#'  -dir_diff/dist to nearest teammate
#'  -dir_diff/dist to nearest opponent
#'  
#'  -time elapsed post throw (num_frames post throw/10)
#'  -time until play complete ((max frame - current frame)/10)
#'  
#'  -whether ball_land_xy is close to boundary or not
#'    -sometimes ball_land_xy is out of bounds and players give up trying to catch it
#'    -ball_land_xy dist to out of bounds?
#'  
#'  
#'  TO DO:
#'  velocity - gett velo_x, velo_y then convert it to the direction the player is heading (vector of speed)
#'  velocity to ball
#'  acceleration to ball
#'  
#'  
#'  -speed of nearest offensive player
#'  -direction of nearest offensive player (the defense is trying to copy/predict this)
#'  -direction to quarterback
#'  
#'  -velo_to ball
#'  -acc_to_ball
#'  
#'  
#'  prev_speed
#'  prev_acc
#'  

#'  
#'  -include acceleration vector and acceleration scalar (the scalar we can use recorded values)
#'  
#'  -update ball_land_dir_diff - update to be the nearest direction to a circle around ball land x,y 
#'                               player just needs to be in this radius to catch, not exactly on the ball land x,y point
#'                               
#'  -avg_dir_offense - the average direction the offense is headed (maybe only take average of people close to ball_land?)
#'  -speed_of_offense- same as above but speed
#'  
#'  
#'  -Voronoi features - this captures the space which players are controlling in the field 


#derive most features here
train_derived = train %>%
  est_kinematics() %>% 
  change_in_kinematics() %>%
  derived_features()

#dist ball land out
dist_ball_land_out = dist_ball_land_out(train)

train_derived = left_join(train_derived, dist_ball_land_out, by = "game_play_id")


#get remaining closest player features
#this takes the longest
#only do it on prop_play_complete >= 0.3 since thats at least what we train the models on

#just derive the close features here and join them back rather than applying it to the entire train df
train_close_features = train %>% filter(prop_play_complete >= 0.3) %>% #dont need to derive on start of play
  select(game_play_id, game_player_play_id, player_side, player_to_predict, frame_id, x, y, est_dir)

plan(sequential) #quit any existing parallel workers
start = Sys.time()
close_player_features = train_close_features %>% 
  #filter(game_play_id %in% 1:100) %>% #for testing speed
  closest_player_dist_dir()
end = Sys.time()
end-start
plan(sequential) #quit parallel workers

write.csv(close_player_features, file = here("data", "closest_dir_dist_features.csv"), row.names = FALSE)


#join the closest features to train_derived
train_derived = train_derived %>%
  full_join(close_player_features, by = c("game_player_play_id", "frame_id"))

#filter only player_to_predict
#we don't need the others for modelling
train_derived = train_derived %>% filter(player_to_predict)

colnames(train_derived)
dim(train_derived)

#save cleaned data
#write.csv(train_derived, file = here("data", "train_clean.csv"), row.names = FALSE)






############################################### Continue EDA ############################################### 

#train = read.csv(file = here("data", "train_clean.csv"), row.names = FALSE)





#ball_land_dir_diff vs fut_dir_diff
ball_land_dir_diff_v_fut_dir_diff = ggplot(data = train %>% filter(throw == "post"), mapping = aes(x = ball_land_dir_diff, y = fut_dir_diff)) + 
  geom_scattermore(alpha = 0.01) + ylim(c(-30,30))
ball_land_dir_diff_v_fut_dir_diff
#' negative relationship here makes sense
#' if the difference in direction to where you're going and where you need to go is negative, then you should start heading positive 
#' 
#' relationship gets even stronger if you filter out first bit of play - prop_play_complete >= 0.4

#out_bounds_dir_diff vs fut_dir_diff
ggplot(data = train, mapping = aes(x = out_bounds_dir_diff, y = fut_dir_diff)) +
  geom_scattermore(alpha = 0.01) + ylim(c(-30, 30))

#closest_player_dir_diff vs fut_dir_diff
train %>% filter(player_side == "Offense") %>% ggplot(mapping = aes(x = closest_opponent_dir_diff, y = fut_dir_diff)) +
  geom_scattermore(alpha = 0.05) + ylim(c(-30, 30))

#closest_player_dir_diff vs fut_dir_diff
train %>% filter(player_side == "Defense") %>% ggplot(mapping = aes(x = closest_teammate_dir_diff, y = fut_dir_diff)) +
  geom_scattermore(alpha = 0.05) + ylim(c(-30, 30))

train %>% filter(player_side == "Defense") %>% ggplot(mapping = aes(x = closest_opponent_dir_diff, y = fut_dir_diff)) +
  geom_scattermore(alpha = 0.05) + ylim(c(-30, 30))

#' speed throughout play
train_derived %>% 
  ggplot(mapping = aes(x = prop_play_complete, y = est_speed)) +
  geom_scattermore(alpha = 0.1) + theme_bw()
#speed gets higher throughout play

#' acc vs pre-post throw
train_derived %>% 
  ggplot(mapping = aes(x = prop_play_complete, y = est_acc)) +
  geom_scattermore(alpha = 0.1) + theme_bw() + ylim(c(-12, 12))
#acceleration more mixed, slightly higher pre throw actually
#makes sense, the player is usually running full speed after ball is in the air

#' ball_land_dir_diff vs pre-post throw
train_derived %>%
  ggplot(mapping = aes(x = prop_play_complete, y = abs(ball_land_dir_diff))) +
  geom_scattermore(alpha = 0.05) + theme_bw()
#difference in direction to ball land smaller post throw, makes sense

#' speed vs direction change
train_derived %>%
  ggplot(mapping = aes(x = prev_dir_diff, y = est_speed)) +
  geom_scattermore() +
  theme_bw() 

#' when players are changing direction, they are going slower
#' some speed values here are clearly outliers potentially impossible - investigate these

#' acceleration vs direction change
train_derived %>%
  ggplot(mapping = aes(x = prev_dir_diff, y = est_acc)) +
  geom_scattermore(alpha = 0.05) +
  scale_y_continuous(limits = c(-25, 25)) +
  theme_bw() 
#' same as above, when direction changing, acceleration is going to 0
#' this is kind of counter-intuitive
#' also some outliers here, look into those, I'm guessing its the values right at the start



#' generally, as play progresses the player's direction diff to ball land x,y gets closer to 0
#' as the player gets super close to the ball though things get weird since small changes in distances give big changes in angles, but doesn't matter much
#' therefore this feature is more important at farther distances, less important at closer distances
#' interaction with current distance to ball

#' first do some EDA to see how close player needs to get to catch the ball, or how close they usually get


#' ball_land_dir_diff vs speed
train_derived %>% 
  ggplot(mapping = aes(x = cut(ball_land_dir_diff, breaks = c(0, 45, 90, 135, 180)), y = est_speed)) +
  geom_boxplot() +
  labs(x = "Ball Land Direction Difference", y = "Speed") + 
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
  ggplot(mapping = aes(x = cut(dist_ball_land, breaks = c(0, 10, 20, 30, 40, 50, 60)), y = est_acc)) +
  geom_boxplot() +
  labs(x = "Distance to ball landing location", y = "Acceleration") +
  #geom_scattermore(alpha = 0.05) +
  scale_y_continuous(limits = c(0, 20)) +
  theme_bw()
#acceleration higher when farther from ball, makes sense


#also splitting this up by offense and defense makes sense - they all have different relationships, since they have different goals...



#closest_features 

data_mod %>% filter(player_side == "Offense") %>% ggplot(mapping = aes(x = closest_opponent_dir_diff, y = fut_dir_diff)) +
  geom_scattermore(alpha = 0.05) +
  ylim(c(-30, 30))

data_mod %>% filter(player_side == "Defense") %>% ggplot(mapping = aes(x = closest_teammate_dir_diff, y = fut_dir_diff)) +
  geom_scattermore(alpha = 0.05) +
  ylim(c(-30, 30))

data_mod %>% filter(player_side == "Defense") %>% ggplot(mapping = aes(x = closest_opponent_dir_diff, y = fut_dir_diff)) +
  geom_scattermore(alpha = 0.05) +
  ylim(c(-30, 30))



data_mod %>% filter(player_side == "Offense") %>% ggplot(mapping = aes(x = closest_opponent_dist, y = fut_s)) +
  geom_scattermore(alpha = 0.1) + xlim(c(0, 20))

data_mod %>% filter(player_side == "Defense") %>% ggplot(mapping = aes(x = closest_opponent_dist, y = fut_s)) +
  geom_scattermore(alpha = 0.05)

data_mod %>% filter(player_side == "Defense") %>% ggplot(mapping = aes(x = closest_opponent_dist, y = fut_s)) +
  geom_scattermore(alpha = 0.05)  + xlim(c(0, 20))





############################################### Export Derived Dataset for Modelling ############################################### 


#manipulating df for future modelling
data_mod = train %>% 
  #order the columns
  select(fut_dir_diff, fut_s_diff, fut_a_diff, game_player_play_id, game_play_id, everything()) %>%
  #de-select unnecessary feature columns - things that can't be calculated post throw
  select(-c(game_id, nfl_id, play_id, player_to_predict, o, s, a, dir, num_frames)) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(est_speed = ifelse(est_speed == 0, 0.01, est_speed)) %>% #0.01 is the min recorded/estimated speed
  group_by(game_player_play_id) %>%
  mutate(fut_s = lead(est_speed),
         fut_a = lead(est_acc)) %>%
  ungroup()

#what proportion of play being complete is ball thrown
data_mod %>% filter(throw == "post" & lag(throw) == "pre") %>% pull(prop_play_complete) %>% hist()
#min is 0.225 - fit models mostly on post-throw
#prop_play_cutoff = 0.4 #this is the cutoff for the amount of play being complete we will fit the model on
#set at 40% play completion right now

#data_mod = data_mod %>% 
#  filter((throw == "post" | throw == "pre" & lead(throw) == "post") | prop_play_complete >= 0.4)


#filter out the dir,s,a diffs in training set that are clearly impossible
data_mod %>% filter(throw == "post") %>% select(fut_dir_diff, fut_s_diff, fut_s, fut_a_diff, fut_a) %>% summary()

#histograms of response
data_mod %>% filter(throw == "post") %>% pull(est_dir) %>% hist()
data_mod %>% filter(throw == "post") %>% pull(est_speed) %>% hist(breaks = 100)
data_mod %>% filter(throw == "post") %>% pull(est_acc) %>% hist(xlim = c(-30, 30), breaks = 500)

data_mod %>% filter(throw == "post") %>% pull(fut_dir_diff) %>% hist(xlim = c(-60, 60), breaks = 200)
data_mod %>% filter(throw == "post") %>% pull(fut_s_diff) %>% hist(breaks = 500, xlim = c(-2, 2))
data_mod %>% filter(throw == "post") %>% pull(fut_a_diff) %>% hist(xlim = c(-30, 30), breaks = 500)

data_mod %>% filter(throw == "post") %>% pull(fut_dir_diff) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)
data_mod %>% filter(throw == "post") %>% pull(fut_s_diff) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)
data_mod %>% filter(throw == "post") %>% pull(fut_s) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)
data_mod %>% filter(throw == "post") %>% pull(fut_a_diff) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)

#save
#write.csv(data_mod, file = here("data", "data_mod_no_close_player.csv"), row.names = FALSE)












