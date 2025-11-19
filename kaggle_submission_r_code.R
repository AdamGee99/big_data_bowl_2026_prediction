############################################### Helpers ###############################################
library(tidyverse)
library(here)
library(catboost)


#' function to calculate direction between two frames
#' direction in degrees and on the coordinate system used by BDB (0 deg is vertical, clockwise)
#' inputs: x_diff, y_diff between current and previous frame
get_dir = function(x_diff, y_diff) {
  (90 - (atan2(y = y_diff, x = x_diff)*180/pi)) %% 360 
}
#be careful when there's no change in position (x_diff = y_diff = 0) - returns 90


#' function to calculate the distance traveled between two frames
#' inputs: distance travelled in x, y coordinates
get_dist = function(x_diff, y_diff) {
  sqrt(x_diff^2 + y_diff^2)
}

#' function that returns the minimum positive or negative direction between two direction differences
min_pos_neg_dir = function(dir_diff) {
  pos_diff = dir_diff %% 360
  neg_diff = -dir_diff %% 360
  
  ifelse(pos_diff <= neg_diff, pos_diff, -neg_diff)
}



############################################### Feature Derivation Functions ############################################### 


#' function that takes in train df and estimates dir, s, and a
#' df needs to have columns: player_to_predict, game_player_play_id, x, y
#' the x,y here can be either the true values or the predicted values
#' right now just using the previous frame only (lag = 1, lead = 0) to calculate
est_kinematics = function(df) {
  kin_df = df %>%
    group_by(game_player_play_id) %>%
    #calculate s, a, dir from true values of x,y values
    mutate(est_speed = get_dist(x_diff = x - lag(x), y_diff = y - lag(y))/0.1, #speed over previous -> current frame
           est_acc = (est_speed - lag(est_speed))/0.1, #acc over previous -> current frame (has direction)
           est_dir = get_dir(x_diff = x - lag(x), y_diff = y - lag(y))) %>%
    mutate(est_speed = ifelse(est_speed == 0 & throw == "post", 0.01, est_speed)) %>% #no 0 speed values post throw
    mutate(est_speed = ifelse(throw == "post", est_speed, s), #use true recorded values pre throw if possible
           est_dir = ifelse(throw == "post", est_dir, dir)) %>%
    ungroup() 
  
  kin_df
}

#' function that takes in training df and gets the previous and future change in dir, s, a
#' the prev change can be used as a feature
#' the future change is the response
#' the input df needs to have kinematics (dir, s, a) for this function to work
change_in_kinematics = function(df) {
  change_kin_df = df %>%
    group_by(game_player_play_id) %>%
    mutate(prev_dir_diff = min_pos_neg_dir(est_dir - lag(est_dir)), #diff in dir from previous -> current frame
           prev_a_diff = est_acc - lag(est_acc) #diff in acc from previous -> current frame
    ) %>%
    ungroup()
  
  change_kin_df
}

#' function that takes in df and calculates all our derived features
derived_features = function(df) {
  derived_df = df %>%
    mutate(curr_ball_land_dir = get_dir(x_diff = ball_land_x - x, y_diff = ball_land_y - y), #direction needed from current point to go to the ball landing point
           ball_land_dir_diff = min_pos_neg_dir(est_dir - curr_ball_land_dir), #difference in current direction of player and direction needed to go to to reach ball land (x,y)
           dist_ball_land = get_dist(x_diff = ball_land_x - x, y_diff = ball_land_y - y), #the distance where the player currently is to where the ball will land
           #distance to closest out of bounds point
           out_bounds_dist = case_when( 
             ((x - 0) <= (120 - x)) & ((x - 0) <= (53.3 - y)) & ((x - 0) <= (y - 0)) ~ x - 0,
             ((120 - x) <= (53.3 - x)) & ((120 - x) <= (y - 0)) ~ 120 - x,
             ((53.3 - y) <= (y - 0)) ~ 53.3 - y,
             .default = y - 0
           ),
           #direction to closest out of bounds point (always 0, 90, 180, 270)
           out_bounds_dir = case_when(
             out_bounds_dist == (x - 0) ~ 270,
             out_bounds_dist == (120 - x) ~ 90,
             out_bounds_dist == (53.3 - y) ~ 0,
             out_bounds_dist == (y - 0) ~ 180
           ),
           out_bounds_dir_diff = min_pos_neg_dir(est_dir - out_bounds_dir),
    ) %>%
    select(-c(out_bounds_dir, curr_ball_land_dir)) 
  
  derived_df
}




############################################### Predict Function ############################################### 


test_input = read.csv(here("data", "test_input.csv")) 
test = read.csv(here("data", "test.csv"))
#add game_play and game_player_play ids
test_input = test_input %>%
  mutate(player_to_predict = as.logical(player_to_predict),
         throw = "pre") %>%
  group_by(game_id, play_id) %>%
  mutate(game_play_id = cur_group_id(),
         prop_play_complete = frame_id/(max(frame_id) + num_frames_output)) %>%
  group_by(game_id, nfl_id, play_id) %>%
  mutate(game_player_play_id = cur_group_id()) %>%
  ungroup() %>%
  filter(player_to_predict) #only on player_to_predict

#select final frame pre throw and add features
data_mod_test_pred = test_input %>% 
  group_by(game_player_play_id) %>% 
  slice_tail(n = 4) %>% #need fourth to last frame here to calculate lag stuff
  arrange(game_player_play_id, frame_id) %>%
  #add features
  est_kinematics() %>%
  mutate(est_speed = s, #use recorded values for dir, s
         est_dir = dir) %>% 
  change_in_kinematics() %>%
  derived_features() %>%
  #keep only last frame pre throw
  group_by(game_player_play_id) %>%
  slice_tail(n = 1) %>%
  ungroup()

#the plays we need to predict on
test_plays = data_mod_test_pred$game_play_id %>% unique() %>% sort()
#features for catboost
cat_features = c( "throw", "play_direction", "absolute_yardline_number", "player_height",
                  "player_weight", "player_position", "prop_play_complete", "est_speed", 
                  "est_acc", "est_dir", "prev_dir_diff", "prev_a_diff", "ball_land_dir_diff",
                  "dist_ball_land", "out_bounds_dist", "out_bounds_dir_diff")

#load in models - these are the final models we get after cv eventually
dir_cat_o = catboost.load_model(model_path = here("models", paste0("fold_", 1), "offense", "dir.cbm"))
dir_cat_d = catboost.load_model(model_path = here("models", paste0("fold_", 1), "defense", "dir.cbm"))
speed_cat_o = catboost.load_model(model_path = here("models", paste0("fold_", 1), "offense", "speed.cbm"))
speed_cat_d = catboost.load_model(model_path = here("models", paste0("fold_", 1), "defense", "speed.cbm"))
acc_cat_o = catboost.load_model(model_path = here("models", paste0("fold_", 1), "offense", "acc.cbm"))
acc_cat_d = catboost.load_model(model_path = here("models", paste0("fold_", 1), "defense", "acc.cbm"))

#now predict through all num_frames_output

#' flow of this:
#'  -loop through all the plays
#'    -loop through all the frames in a play
#'    -for each player: predict next x,y and derive new features
#'      -loop through the players in the play post throw
#'      -predict next dir, s, a

#storage
results_list = list()
#loop through all plays
for(play in test_plays) {
  #info for current play
  curr_play_info = data_mod_test_pred %>% filter(game_play_id == play) 
  
  #number of frames to predict
  num_frames_output = curr_play_info$num_frames_output %>% unique()
  #players on this play
  player_ids = curr_play_info$game_player_play_id
  #the last frame_id pre throw 
  last_frame_id = curr_play_info$frame_id %>% unique()
  
  #loop through frames to predict on
  for(output_frame_id in 1:num_frames_output) {
    frame = last_frame_id + output_frame_id
    
    #info for all players in current frame
    curr_frame_all_players = curr_play_info %>% 
      mutate(frame_id = frame, #add proper frame_id stuff
             prop_play_complete = frame_id/(last_frame_id + num_frames_output))
    
    #update throw to be post
    if (output_frame_id > 1) {curr_frame_all_players$throw = "post"}
    
    #if frame is pre throw we already know the features and everything to predict x,y,dir,s,a
    
    #if frame is post throw then we need to update position and dir, s, a based on previous iterations predictions
    if (unique(curr_frame_all_players$throw) == "post") { 
      #update current x, y dir, s, a as previous prediction
      prev_frame_all_players = result
      curr_frame_all_players = curr_frame_all_players %>%
        mutate(x = prev_frame_all_players$pred_x,
               y = prev_frame_all_players$pred_y,
               est_dir = prev_frame_all_players$pred_dir,
               est_speed = prev_frame_all_players$pred_s,
               est_acc = prev_frame_all_players$pred_a,
               #initialize the preds to be NA
               pred_x = NA,
               pred_y = NA,
               pred_dir = NA,
               pred_s = NA,
               pred_a = NA) 
      
      #derive features
      
      #need previous frame to compute lag stuff
      prev_curr_frame_df = curr_frame_all_players %>%
        rbind(prev_frame_all_players) %>%
        arrange(game_player_play_id, frame_id)
      
      #derive other features
      curr_frame_all_players = prev_curr_frame_df %>%
        group_by(game_player_play_id) %>% #make sure to group_by each player
        change_in_kinematics() %>%
        derived_features() %>%
        filter(frame_id == frame) #filter for only current frame
    }
    
    #predict next x,y
    curr_frame_all_players = curr_frame_all_players %>%
      mutate(pred_dist_diff = est_speed*0.1 + est_acc*0.5*0.1^2,
             pred_x = x + cos(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff,
             pred_y = y + sin(((90 - est_dir) %% 360)*pi/180)*pred_dist_diff) %>%
      select(-pred_dist_diff)
    
    #predict next dir, s, a
    #need to loop through each player since catboost load pool thing
    curr_frame_all_players_pred = list()
    for(player in player_ids) {
      cat_pred_df = curr_frame_all_players %>%
        filter(game_player_play_id == player) %>%
        select(all_of(cat_features)) %>% #pred df for catboost
        mutate(across(where(is.character), as.factor)) %>%
        catboost.load_pool()
      
      #return row with predicted dir, s, a
      curr_frame_player = curr_frame_all_players %>% 
        filter(game_player_play_id == player) #predict on single player
      
      #load in models
      if(unique(curr_frame_player$player_side) == "Offense") {
        dir_cat = dir_cat_o
        speed_cat = speed_cat_o
        acc_cat = acc_cat_o
      } else {
        dir_cat = dir_cat_d
        speed_cat = speed_cat_d
        acc_cat = acc_cat_d
      }
      #predict
      curr_frame_player = curr_frame_player %>%  
        mutate(pred_dir = est_dir + catboost.predict(dir_cat, cat_pred_df),
               pred_s = catboost.predict(speed_cat, cat_pred_df), 
               pred_s = exp(pred_s), #exponentiate back to original scale
               pred_a = est_acc + catboost.predict(acc_cat, cat_pred_df))
      
      #same colnames as previous iteration
      if(curr_frame_player$throw != "pre") {curr_frame_all_players = curr_frame_all_players %>% select(all_of(colnames(result)))} 
      
      #append to list
      curr_frame_all_players_pred[[paste0(play, "_", player, "_", frame)]] = curr_frame_player
    }
    
    ### store result for this frame ###
    result = curr_frame_all_players_pred %>% bind_rows()
    results_list[[paste0(play, "_", frame)]] = result
  }
}
#back into df
results_pred = results_list %>%
  bind_rows() %>% 
  arrange(game_play_id, game_player_play_id, frame_id)

#return predictions
results_pred %>% 
  select(pred_x, pred_y) %>%
  rename(x = pred_x, y = pred_y)



