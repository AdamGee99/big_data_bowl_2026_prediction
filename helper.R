#various helper functions

library(tidyverse)
library(here)
library(scales)
library(patchwork)



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

#' same as above but function takes in a vector
get_dist_vector = function(diffs) {
  sqrt(diffs[1]^2 + diffs[2]^2)
}


#' function to get rmse (evaluation metriic) for predictions
#' inputs: true x,y values, predicted x,y values
#' true and predicted vectors must be same length
get_rmse = function(true_x, true_y, pred_x, pred_y) {
  n = length(true_x)
  
  x_diff = pred_x - true_x
  y_diff = pred_y - true_y
  
  rmse = sqrt(1/(2*n)*sum(x_diff^2 + y_diff^2))
  rmse
}

#' function that returns the minimum positive or negative direction between two direction differences
min_pos_neg_dir = function(dir_diff) {
  pos_diff = dir_diff %% 360
  neg_diff = -dir_diff %% 360
  
  ifelse(pos_diff <= neg_diff, pos_diff, -neg_diff)
}


#' takes in a dataframe for all players in the same frame
#' outputs the min distance and direction to the closest other player
#' input df must have: game_player_play_id, player_side, est_dir, x, y for each player in the frame
get_closest_player_min_dist_dir = function(df) {
  if(nrow(df) == 1) { #if only one player to predict post throw - NA closest player features
    data.frame(game_player_play_id = df$game_player_play_id,
               closest_teammate_dist = NA,
               closest_teammate_dir_diff = NA,
               closest_opponent_dist = NA,
               closest_opponent_dir_diff = NA)
  } else {
    pairs = combn(nrow(df), 2, FUN = function(id) {
      i = id[1]
      j = id[2]
      
      data.frame(
        player1 = df$game_player_play_id[i],
        id1 = df$game_player_play_id[i],
        player2 = df$game_player_play_id[j],
        id2 = df$game_player_play_id[j],
        other_player_distance = get_dist(df$x[j] - df$x[i], df$y[j] - df$y[i]),
        other_player_dir = get_dir(df$x[j] - df$x[i], df$y[j] - df$y[i]),
        player_side = df$player_side[i],
        other_player_side = df$player_side[j],
        player_dir = df$est_dir[i]
      )
    }, simplify = FALSE) %>% bind_rows()  %>% 
      pivot_longer(cols = c(player1, player2),
                   names_to = "role",
                   values_to = "player1") %>%
      mutate(player2 = ifelse(role == "player1", id2, id1)) %>% 
      select(-c(id1, id2, role))
    pairs
    
    #closest teammate
    closest_teamate = pairs %>%
      filter(player_side == other_player_side) %>% 
      group_by(player1) %>%
      mutate(distance = min(other_player_distance)) %>% 
      filter(distance == other_player_distance) %>%
      mutate(closest_teammate_dir_diff = min_pos_neg_dir(player_dir - other_player_dir)) %>% 
      rename(closest_teammate_dist = distance, game_player_play_id = player1) %>%
      select(game_player_play_id, closest_teammate_dist, closest_teammate_dir_diff)
    
    #closest opponent
    closest_opponent =  pairs %>%
      filter(player_side != other_player_side) %>% 
      group_by(player1) %>%
      mutate(distance = min(other_player_distance)) %>% 
      filter(distance == other_player_distance) %>%
      mutate(closest_opponent_dir_diff = min_pos_neg_dir(player_dir - other_player_dir)) %>% 
      rename(closest_opponent_dist = distance, game_player_play_id = player1) %>%
      select(game_player_play_id, closest_opponent_dist, closest_opponent_dir_diff)
    
    full_join(closest_teamate, closest_opponent, by = "game_player_play_id")
  }
}

# 
# ### SOMETHING IS WRONG HERE - ITS NOT CALCULATING CLOSEST_OPP_DIR_DIFF PROPERLY
# all_player_curr_frame = train %>% 
#   filter(game_play_id == 6, frame_id == 24) %>%
#   select(game_player_play_id, game_play_id, frame_id, x, y, est_dir, player_side)



############################################### Feature Derivation Functions ############################################### 


#' these functions are intended to be used in the order listed here


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
    mutate(est_speed = ifelse(throw == "post", est_speed, s), #use true recorded values pre throw if possible
           est_dir = ifelse(throw == "post", est_dir, dir),
           max_frame_id = max(frame_id)) %>% #the maximum frame id for this player on this play
    ungroup() 
  
  kin_df
}

#' function that takes in training df and gets the previous and future change in dir, s, a
#' the prev change can be used as a feature
#' the future change is the response
#' 
#' the input df needs to have kinematics (dir, s, a) for this function to work
change_in_kinematics = function(df) {
  change_kin_df = df %>%
    group_by(game_player_play_id) %>%
    mutate(prev_dir_diff = min_pos_neg_dir(est_dir - lag(est_dir)), #diff in dir from previous -> current frame
           prev_a_diff = est_acc - lag(est_acc), #diff in acc from previous -> current frame
           fut_dir_diff = min_pos_neg_dir(lead(est_dir) - est_dir), #diff in dir from current -> next frame
           fut_s_diff = lead(est_speed) - est_speed, #diff in speed from current -> next frame
           fut_a_diff = lead(est_acc) - est_acc #diff in acc from current -> next frame
           ) %>%
    ungroup()
  
  change_kin_df
}


#' function that adds distance and direction to nearest teammate and opposing player
closest_player_dist_dir = function(df) {
  game_play_ids = df$game_play_id %>% unique()
  
  closest_results = foreach(game_play = game_play_ids, .combine = rbind, .packages = c("tidyverse", "doParallel", "here")) %dopar% { #loop through the plays
    source(here("helper.R"))
    curr_game_play = df %>% filter(game_play_id == game_play)
    frames = curr_game_play$frame_id %>% unique()
    
    #loop through the frames in a play
    foreach(frame = frames, .combine = rbind) %do% {
      #all players in this frame
      curr_frame_all_players = curr_game_play %>% 
        filter(frame_id == frame) 
      
      #get closest player features for players to predict
      closest_features = curr_frame_all_players %>%
        filter(player_to_predict) %>%
        get_closest_player_min_dist_dir()
      
      #join the closest features back
      curr_frame_all_players = curr_frame_all_players %>%
        full_join(closest_features, by = "game_player_play_id")
      curr_frame_all_players
    }
  }
  #get direction diff
  closest_results = closest_results %>%
    arrange(game_play_id, game_player_play_id, frame_id) #arrange in right order
  closest_results
}



#' function that takes in df and calculates all our derived features
derived_features = function(df) {
  derived_df = df %>%
    mutate(curr_ball_land_dir = get_dir(x_diff = ball_land_x - x, y_diff = ball_land_y - y), #direction needed from current point to go to the ball landing point
           ball_land_dir_diff = min_pos_neg_dir(est_dir - curr_ball_land_dir), #difference in current direction of player and direction needed to go to to reach ball land (x,y)
           dist_ball_land = get_dist(x_diff = ball_land_x - x, y_diff = ball_land_y - y), #the distance where the player currently is to where the ball will land
           time_elapsed = frame_id*0.1, #time elapsed in seconds
           time_until_play_complete = (max_frame_id - frame_id)*0.1, #time until play complete in seconds
           
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
           #distance to closest player
           
           #direction to closest player
           
           ) %>%
    select(-c(out_bounds_dir, curr_ball_land_dir)) 
    
  derived_df
}







############################################### Plotting functions ############################################### 






#' plotting a player's speed and acceleration over a play
#' calculation depends on number of lagging and leading frames
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
           a_scalar = ifelse(is.na(a), acc_scalar, a),
           a_vector = ifelse(is.na(a), acc_vector, a)) %>%
    select(-c(speed, acc_scalar, acc_vector))
  
  #make speed plot
  speed_plot = ggplot(data = plot_df, mapping = aes(x = frame_id, y = s, colour = obs)) +
    geom_point() +
    theme_bw() +
    labs(x = "", y = "Speed (yds/sec)", title = "Speed") +
    theme(legend.title = element_blank(),
          legend.position = "bottom")
  
  #make acceleration plot
  acc_plot =  ggplot(data = plot_df, mapping = aes(x = frame_id, y = a_scalar, colour = obs)) +
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

#' plot player's observed vs estimated direction over a play
#' calculation depends on number of ledaing and lagging frames
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
           est_dir = get_dir(x_diff = x_diff, y_diff = y_diff)) %>%
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


#' visualise single player's movement
#' function only plots players to predict
plot_player_movement = function(group_id) {
  plot_df = train %>% 
    filter(player_to_predict) %>% #filter for only players to predict
    group_by(game_player_play_id) %>% 
    filter(cur_group_id() == group_id) %>% #filter for a single player on a single play
    select(frame_id, game_id, nfl_id, play_id, x, y, ball_land_x, ball_land_y, throw, player_side) %>%
    ungroup() 
  
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  nfl_id = plot_df$nfl_id %>% unique()
  
  #plot
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = frame_id, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_gradient(low = "black", high = "green") +
    geom_point(aes(fill = "True"), x = NA, y = NA, color = "green", size = 2.5) + #dummy variable for legend
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y, fill = "Ball Land"), colour = "red", size = 4) +
    #geom_point(mapping = aes(x = pred_x, y = pred_y, fill = "Predicted"), colour = "orange", size = 2.5) +
    scale_fill_manual(name = "", values = c("True" = 16, "Ball Land" = 16, "Predicted" = 16)) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id, ", Player ID: ", nfl_id)) +
    theme_bw() +
    guides(colour = "none", shape = "none")
}

#same as above but plotting the actual game_player_play_id
plot_player_movement_game_player_play_id = function(group_id) {
  plot_df = train %>% 
    filter(game_player_play_id == group_id) %>% #filter for only players to predict
    select(frame_id, game_id, nfl_id, play_id, x, y, ball_land_x, ball_land_y, throw, player_side) %>%
    ungroup() 
  
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  nfl_id = plot_df$nfl_id %>% unique()
  
  #plot
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = frame_id, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_gradient(low = "black", high = "green") +
    geom_point(aes(fill = "True"), x = NA, y = NA, color = "green", size = 2.5) + #dummy variable for legend
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y, fill = "Ball Land"), colour = "red", size = 4) +
    #geom_point(mapping = aes(x = pred_x, y = pred_y, fill = "Predicted"), colour = "orange", size = 2.5) +
    scale_fill_manual(name = "", values = c("True" = 16, "Ball Land" = 16, "Predicted" = 16)) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id, ", Player ID: ", nfl_id)) +
    theme_bw() +
    guides(colour = "none", shape = "none")
}


#' visualize single player's movement along with prediction
#' need to have predictions first
#' group_id_preds is a df with the frames as a column and pred_x, pred_y as the other two columns
plot_player_movement_pred = function(group_id, group_id_preds) {
  plot_df = train %>% 
    filter(player_to_predict, #filter for only players to predict
           game_player_play_id == group_id) %>% 
    select(frame_id, game_id, nfl_id, play_id, x, y, ball_land_x, ball_land_y, throw, player_side) %>%
    #join predictions
    left_join(group_id_preds, by = "frame_id")
  
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  nfl_id = plot_df$nfl_id %>% unique()
  
  #plot
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = frame_id, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_gradient(low = "black", high = "green") +
    geom_point(aes(fill = "True"), x = NA, y = NA, color = "green", size = 2.5) + #dummy variable for legend
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y, fill = "Ball Land"), colour = "red", size = 4) +
    geom_point(mapping = aes(x = pred_x, y = pred_y, fill = "Predicted"), colour = "orange", size = 2.5) +
    scale_fill_manual(name = "", values = c("True" = 16, "Ball Land" = 16, "Predicted" = 16)) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id, ", Player ID: ", nfl_id)) +
    theme_bw() +
    guides(colour = "none", shape = "none")
}


#' function now plotting multiple players on same play
multi_player_movement = function(group_id, only_player_to_predict = TRUE) {
  if (only_player_to_predict) {
  plot_df = train %>% 
    filter(player_to_predict) %>% #filter for only players that were targeted
    group_by(game_id, play_id) %>% 
    filter(cur_group_id() == group_id) %>% #filter for a single play
    select(game_id, play_id, frame_id, x, y, ball_land_x, ball_land_y, throw, player_side, player_position, player_name) %>%
    mutate(colour = case_when(
      player_side == "Defense" ~ col_numeric(c("black", "blue"), domain = range(frame_id))(frame_id),
      player_side == "Offense" & player_position == "QB" ~ col_numeric(c("black", "red"), domain = range(frame_id))(frame_id),
      .default =  col_numeric(c("black", "green"), domain = range(frame_id))(frame_id)
     )) %>% #add colours depending on offense or defense
    ungroup()
  } else {
    plot_df = train %>% 
    group_by(game_id, play_id) %>% 
    filter(cur_group_id() == group_id) %>% #filter for a single play
    select(game_id, play_id, frame_id, x, y, ball_land_x, ball_land_y, throw, player_side, player_position, player_name) %>%
    mutate(colour = case_when(
      player_side == "Defense" ~ col_numeric(c("black", "blue"), domain = range(frame_id))(frame_id),
      player_side == "Offense" & player_position == "QB" ~ col_numeric(c("black", "red"), domain = range(frame_id))(frame_id),
      .default =  col_numeric(c("black", "green"), domain = range(frame_id))(frame_id)
     )) %>% #add colours depending on offense or defense
        ungroup()
  }
  
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  
  #plot 
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = colour, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_identity() + 
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y), colour = "red", size = 4) +
    scale_x_continuous(n.breaks = 10) +
    scale_y_continuous(n.breaks = 10) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id)) +
    theme_bw() +
    guides(shape = "none")
}



#' same as above but group_id is game_play_id
multi_player_movement_game_play_id = function(group_id, only_player_to_predict = TRUE) {
  if (only_player_to_predict) {
    plot_df = train %>% 
      filter(game_play_id == group_id) %>% #filter for only players that were targeted
      select(game_id, play_id, frame_id, x, y, ball_land_x, ball_land_y, throw, player_side, player_position, player_name) %>%
      mutate(colour = case_when(
        player_side == "Defense" ~ col_numeric(c("black", "blue"), domain = range(frame_id))(frame_id),
        player_side == "Offense" & player_position == "QB" ~ col_numeric(c("black", "red"), domain = range(frame_id))(frame_id),
        .default =  col_numeric(c("black", "green"), domain = range(frame_id))(frame_id)
      )) %>% #add colours depending on offense or defense
      ungroup()
  } else {
    plot_df = train %>% 
      filter(game_play_id == group_id) %>% #filter for a single play
      select(game_id, play_id, frame_id, x, y, ball_land_x, ball_land_y, throw, player_side, player_position, player_name) %>%
      mutate(colour = case_when(
        player_side == "Defense" ~ col_numeric(c("black", "blue"), domain = range(frame_id))(frame_id),
        player_side == "Offense" & player_position == "QB" ~ col_numeric(c("black", "red"), domain = range(frame_id))(frame_id),
        .default =  col_numeric(c("black", "green"), domain = range(frame_id))(frame_id)
      )) %>% #add colours depending on offense or defense
      ungroup()
  }
  
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  
  #plot 
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = colour, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_identity() + 
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y), colour = "red", size = 4) +
    scale_x_continuous(n.breaks = 10) +
    scale_y_continuous(n.breaks = 10) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id)) +
    theme_bw() +
    guides(shape = "none")
}




#' same as above but now with predictions
#' need to have predictions first
#' group_id_preds is a df with the frames as a column and pred_x, pred_y as the other two columns
multi_player_movement_pred = function(group_id, group_id_preds) {
  plot_df = train %>% 
    filter(player_to_predict, #filter for only players that were targeted
           game_play_id == group_id) %>% 
    select(game_id, play_id, game_play_id, game_player_play_id, frame_id, 
           x, y, ball_land_x, ball_land_y, throw, player_side, player_name) %>%
    mutate(colour = ifelse( #add colours depending on offense or defense
      player_side == "Offense", 
      col_numeric(c("black", "green"), domain = range(frame_id))(frame_id),
      col_numeric(c("black", "blue"), domain = range(frame_id))(frame_id)
    )) %>% 
    left_join(group_id_preds, by = c("game_player_play_id", "frame_id"))
  game_id = plot_df$game_id %>% unique()
  play_id = plot_df$play_id %>% unique()
  
  #plot
  ggplot(data = plot_df, mapping = aes(x = x, y = y, colour = colour, shape = throw)) + #plot the movement
    geom_point(size = 3) +
    scale_colour_identity() + 
    geom_point(aes(fill = "True"), x = NA, y = NA, color = "green", size = 2.5) + #dummy variable for legend
    scale_shape_manual(values = c(19, 1)) + #hollow is pre throw, filled is post throw
    geom_point(mapping = aes(x = ball_land_x, y = ball_land_y, fill = "Ball Land"), colour = "red", size = 4) +
    geom_point(mapping = aes(x = pred_x, y = pred_y, fill = "Predicted"), colour = "orange", size = 2.5) +
    scale_fill_manual(name = "", values = c("True" = 16, "Ball Land" = 16, "Predicted" = 16)) +
    labs(title = paste0("Game: ", game_id, ", Play: ", play_id)) +
    scale_x_continuous(n.breaks = 10) +
    scale_y_continuous(n.breaks = 10) +
    theme_bw() +
    guides(colour = "none", shape = "none")
}


#' ball_land_dir_diff vs frame_id
#' the difference in player's current direction to the direction where the ball is landing
plot_ball_land_dir_diff = function(group_id) {
  plot = train_derived %>% 
    group_by(game_id, nfl_id, play_id) %>% 
    filter(cur_group_id() == group_id) %>%
    ungroup() %>%
    ggplot(mapping = aes(x = frame_id, y = ball_land_dir_diff, colour = throw)) +
    geom_point() +
    scale_y_continuous(limits = c(0, 180), breaks = c(0, 90, 180)) +
    labs(x = "Frame ID", y = "Difference in current direction and ball land direction (deg)") +
    theme_bw()  
  
  #plot ball_land_dir_diff and player movement to help visualize
  wrap_plots(list(plot_player_movement(group_id), plot), nrow = 2)
}



#' function that compares prediced vs true dir, s, a for a single player on a play
dir_s_a_eval = function(group_id) {
  curr_game_player_play_id = results_comp %>% 
    group_by(game_player_play_id) %>%
    filter(cur_group_id() == group_id) %>% 
    pull(game_player_play_id) %>% unique()
  
  dir_s_a_eval_df = results_comp %>%
    filter(game_player_play_id == curr_game_player_play_id) %>%
    select(frame_id, est_dir, est_speed, est_acc, true_dir, true_s, true_a)
  
  dir_eval_plot = dir_s_a_eval_df %>% 
    select(frame_id, est_dir, true_dir) %>%
    pivot_longer(cols = -frame_id, names_to = "obs", values_to = "value") %>%
    mutate(obs = ifelse(obs == "est_dir", "Predicted", "True")) %>%
    ggplot(mapping = aes(x = frame_id, y = value, colour = obs)) + 
    geom_line() +
    scale_x_continuous(n.breaks = ceiling(nrow(dir_s_a_eval_df)/2)) +
    scale_colour_manual(values = c("Predicted" = "orange", "True" = "green")) +
    geom_point() + xlab("") + ylab("Direction") +
    theme_bw()
  
  s_eval_plot = dir_s_a_eval_df %>% 
    select(frame_id, est_speed, true_s) %>%
    pivot_longer(cols = -frame_id, names_to = "obs", values_to = "value") %>%
    mutate(obs = ifelse(obs == "est_speed", "Predicted", "True")) %>%
    ggplot(mapping = aes(x = frame_id, y = value, colour = obs)) + 
    geom_line() +
    geom_point() + xlab("") + ylab("Speed") +
    scale_x_continuous(n.breaks = ceiling(nrow(dir_s_a_eval_df)/2)) +
    scale_colour_manual(values = c("Predicted" = "orange", "True" = "green")) +
    theme_bw()
  
  a_eval_plot = dir_s_a_eval_df %>% 
    select(frame_id, est_acc, true_a) %>%
    pivot_longer(cols = -frame_id, names_to = "obs", values_to = "value") %>%
    mutate(obs = ifelse(obs == "est_acc", "Predicted", "True")) %>%
    ggplot(mapping = aes(x = frame_id, y = value, colour = obs)) + 
    geom_line() +
    geom_point() + xlab("Frame ID") + ylab("Acceleration") +
    scale_x_continuous(n.breaks = ceiling(nrow(dir_s_a_eval_df)/2)) +
    scale_colour_manual(values = c("Predicted" = "orange", "True" = "green")) +
    theme_bw()
  
  wrap_plots(list(dir_eval_plot, s_eval_plot, a_eval_plot),
             nrow = 3)  +
    plot_layout(guides = "collect")
}

