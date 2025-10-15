### Libraries ###

library(tidyverse)
library(here)
library(data.table)
library(gridExtra)
library(patchwork)
library(scales)


### Import Data ###

train_X_files = list.files(path = here("data", "train"), pattern = "input")
train_Y_files = list.files(path = here("data", "train"), pattern = "output")

train_X = train_X_files %>% map_dfr(~read.csv(here("data", "train", .x))) #like 5 million rows
train_Y = train_Y_files %>% map_dfr(~read.csv(here("data", "train", .x)))

# these are the pre-throw features
head(train_X)

#these are the post-throw player positions
head(train_Y)


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




### EDA ###


#first lets visualize the player's movements
game_ids = train$game_id %>% unique() #272 games
nfl_ids = train$nfl_id %>% unique() #1384 player ids

#visualise single player's movement

#function only plots players that were targeted
plot_player_movement = function(group_id) {
  train %>% 
    filter(player_to_predict) %>% #filter for only players that were targeted
    group_by(game_id, nfl_id, play_id) %>% 
    filter(cur_group_id() == group_id) %>% #filter for a single player on a single play
    select(frame_id, x, y, ball_land_x, ball_land_y, throw, player_side) %>%
    ungroup() %>% 
    ggplot(mapping = aes(x = x, y = y, colour = frame_id, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_gradient(low = "black", high = "green") +
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y), colour = "red", size = 4) +
    theme_bw() +
    guides(colour = "none", shape = "none")
}

plot_player_movement(group_id = 1)
plot_player_movement(group_id = 2)

#plotting a grid 
plots_1 = lapply(1:9, plot_player_movement)
plots_2 = lapply(10:18, plot_player_movement)
plots_3 = lapply(19:27, plot_player_movement)

grid.arrange(grobs = plots_1, ncol = 3, nrow = 3)
grid.arrange(grobs = plots_2, ncol = 3, nrow = 3)
grid.arrange(grobs = plots_3, ncol = 3, nrow = 3)

num_plots = 16
grid.arrange(grobs = lapply(1:num_plots, plot_player_movement),
             ncol = ceiling(sqrt(num_plots)),
             nrow = ceiling(sqrt(num_plots)))


#function now plotting multiple players on same play
multi_player_movement = function(group_id) {
  train %>% 
    filter(player_to_predict) %>% #filter for only players that were targeted
    group_by(game_id, play_id) %>% 
    filter(cur_group_id() == group_id) %>% #filter for a single play
    select(frame_id, x, y, ball_land_x, ball_land_y, throw, player_side, player_name) %>%
    mutate(colour = ifelse(
      player_side == "Offense", 
      col_numeric(c("black", "green"), domain = range(frame_id))(frame_id),
      col_numeric(c("black", "blue"), domain = range(frame_id))(frame_id)
    )) %>% #add colours depending on offense or defense
    ungroup() %>% 
    ggplot(mapping = aes(x = x, y = y, colour = colour, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_identity() + 
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y), colour = "red", size = 4) +
    theme_bw() +
    guides(shape = "none")
}
multi_player_movement(1)

num_plots = 4
wrap_plots(lapply(1:num_plots, multi_player_movement),
           ncol = ceiling(sqrt(num_plots)),
           nrow = ceiling(sqrt(num_plots))) +
  plot_layout(guides = "collect")


#filtering for all players in play gives interesting results too 


### TO DO ###



#check if theres anything in test thats not in train... (any player for eg)

#certain players are dominant to one side, eg always like to cut right, use this info?

#add speed, direction, acceleration from predictions of model at every frame 

