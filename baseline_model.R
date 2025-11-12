############################################### Libraries ############################################### 

library(tidyverse)
library(here)
library(ggrepel)
library(xgboost)
library(catboost)
library(scattermore)

############################################### Description ############################################### 
#' predicting x,y from direction, speed, acceleration in current frame and updating (predicting) direction, speed, and acceleration in next frame
#' three models that predict change in dir, s, a each
#' separate dir, s, a models for offense and defense 
#' 
#' 
#' 
#' 
#' 
#' DO MORE EDA, figure out if derived features need manipulation
#' also exclude any weird plays on training... or investigate anything else wrong with the data...


############################################### Baseline Model ############################################### 

#import data
train = read.csv(file = here("data", "train_clean.csv"))

source(here("helper.R"))

#' right now only training on players_to_predict, not sure if it makes sense to include the other players?
#' 
#' 
#' THIS IS ALL EDA - MOVE THIS TO ANOTHER FILE NAD ONLY FIT MODELS AND PREDICT HERE

#modelling the future change in dir, s, a
data_mod = train %>% 
  #order the columns
  select(fut_dir_diff, fut_s_diff, fut_a_diff, game_player_play_id, game_play_id, everything()) %>%
  #add estimation to post throw
   mutate(est_dir = ifelse(throw == "pre", dir, est_dir), #use recorded dir, s if possible over estimated...
          est_speed = ifelse(throw == "pre", s, est_speed))  %>% #do not use recorded a since its a scalar, we want vector
  #de-select unnecessary feature columns - things that can't be calculated post throw
  select(-c(game_id, nfl_id, play_id, o, player_to_predict, s, a, dir,
          player_birth_date, num_frames_output, num_frames)) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(est_speed = ifelse(est_speed == 0, 0.01, est_speed)) %>% #0.01 is the min recorded/estimated speed
  mutate(fut_s = lead(est_speed)) %>%
  filter(!is.na(fut_dir_diff) & !is.na(fut_s_diff) & !is.na(fut_a_diff)) #remove NA responses

#what proportion of play being complete is ball thrown
data_mod %>% filter(throw == "post" & lag(throw) == "pre") %>% pull(prop_play_complete) %>% hist()
#min is 0.225 - fit models mostly on post-throw
prop_play_cutoff = 0.4 #this is the cutoff for the amount of play being complete we will fit the model on
#set at 40% play completion right now

data_mod = data_mod %>% 
  filter(throw == "post" | throw == "pre" & lead(throw) == "post" | prop_play_complete >= 0.4)
#mutate 0 est_speeds to be small positive

#' 80% train, 20% test - by play groups
set.seed(1999)
n_plays = data_mod %>% pull(game_play_id) %>% unique() %>% length() #9,109 plays
split = sample(unique(data_mod$game_play_id), size = round(0.8*n_plays))
data_mod_train = data_mod %>% filter(game_play_id %in% split)
data_mod_test = data_mod %>% filter(!(game_play_id %in% split))

#filter out the dir,s,a diffs in training set that are clearly impossible
data_mod %>% filter(throw == "post") %>% select(fut_dir_diff, fut_s_diff, fut_a_diff) %>% summary()

#histograms of response
data_mod %>% filter(throw == "post") %>% pull(fut_dir_diff) %>% hist(breaks = 300, xlim = c(-30, 30))
data_mod %>% filter(throw == "post") %>% pull(fut_s_diff) %>% hist(breaks = 300, xlim = c(-2, 2))
data_mod %>% filter(throw == "post") %>% pull(fut_s) %>% hist(breaks = 200)
data_mod %>% filter(throw == "post") %>% pull(fut_a_diff) %>% hist(breaks = 600, xlim = c(-11, 11))

data_mod %>% filter(throw == "post") %>% pull(fut_dir_diff) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)
data_mod %>% filter(throw == "post") %>% pull(fut_s_diff) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)
data_mod %>% filter(throw == "post") %>% pull(fut_s) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)
data_mod %>% filter(throw == "post") %>% pull(fut_a_diff) %>% quantile(probs = c(0.001, 0.01, 0.99, 0.999), na.rm = TRUE)

#filter out the 1% highest and lowest dir, s and a, these are probably weird data/calculation issues
data_mod_train = data_mod %>% filter(abs(fut_dir_diff) <= 50,
                                     abs(fut_s_diff) <= 1.5,
                                     abs(fut_a_diff) <= 8)

#also get rid of plays that were cleary recorded incorrectly - the plays with way too many frames
num_frames = data_mod %>% filter(throw == "post") %>% group_by(game_player_play_id) %>% summarise(n_frames = n())
num_frames$n_frames %>% hist(breaks = 50)
num_frames$n_frames %>% summary()
num_frames$n_frames %>% quantile(probs = c(0.999))
#identify plays that run too long
long_player_plays = num_frames %>% filter(n_frames >= 50) %>% pull(game_player_play_id)
train %>% filter(game_player_play_id %in% c(long_player_plays)) %>%
  pull(game_play_id) %>% unique()
multi_player_movement_game_play_id(812) #this is wrong

#get rid of this play
data_mod_train = data_mod_train %>% filter(!(game_play_id %in% 812))


#fit models

#drop unnecessary features
unnnecessary_features = c("game_player_play_id", "game_play_id", "player_role",
                          "closest_player_id", "frame_id", "x", "y", 
                          "ball_land_x", "ball_land_y", "player_name")
cat_train_df = data_mod_train %>% select(-all_of(unnnecessary_features))
cat_test_df = data_mod_test %>%select(-all_of(unnnecessary_features))

#offense and defense training sets
train_o = cat_train_df %>% filter(player_side == "Offense") %>% select(-c(starts_with("fut_"), player_side))
train_d = cat_train_df %>% filter(player_side == "Defense") %>% select(-c(starts_with("fut_"), player_side))
train_o_labels = cat_train_df %>% filter(player_side == "Offense") %>% select(c(starts_with("fut_")))
train_d_labels = cat_train_df %>% filter(player_side == "Defense") %>% select(c(starts_with("fut_")))

#test sets
#offense and defense training sets
test_o = cat_test_df %>% filter(player_side == "Offense") %>% select(-c(starts_with("fut_"), player_side))
test_d = cat_test_df %>% filter(player_side == "Defense") %>% select(-c(starts_with("fut_"), player_side))
test_o_labels = cat_test_df %>% filter(player_side == "Offense") %>% select(c(starts_with("fut_")))
test_d_labels = cat_test_df %>% filter(player_side == "Defense") %>% select(c(starts_with("fut_")))


#df in right type for catboost
#train sets
cat_train_dir_o = catboost.load_pool(train_o, label = train_o_labels$fut_dir_diff)
cat_train_dir_d = catboost.load_pool(train_d, label = train_d_labels$fut_dir_diff)
cat_train_speed_o = catboost.load_pool(train_o, label = log(train_o_labels$fut_s)) #try log transform since speed is strictly positive
cat_train_speed_d = catboost.load_pool(train_d, label = log(train_d_labels$fut_s)) #try log transform since speed is strictly positive
cat_train_acc_o = catboost.load_pool(train_o, label = train_o_labels$fut_a_diff)
cat_train_acc_d = catboost.load_pool(train_d, label = train_d_labels$fut_a_diff)
#test sets
cat_test_dir_o = catboost.load_pool(test_o, label = test_o_labels$fut_dir_diff)
cat_test_dir_d = catboost.load_pool(test_d, label = test_d_labels$fut_dir_diff)
cat_test_speed_o = catboost.load_pool(test_o, label = log(test_o_labels$fut_s)) #try log transform since speed is strictly positive
cat_test_speed_d = catboost.load_pool(test_d, label = log(test_d_labels$fut_s)) #try log transform since speed is strictly positive
cat_test_acc_o = catboost.load_pool(test_o, label = test_o_labels$fut_a_diff)
cat_test_acc_d = catboost.load_pool(test_d, label = test_d_labels$fut_a_diff)

#fit - optionally set the test set
dir_cat_o = catboost.train(learn_pool = cat_train_dir_o, test_pool = cat_test_dir_o, params = list(metric_period = 50,
                                                                                                   #iterations = 3000,
                                                                                                   od_type = "Iter", 
                                                                                                   od_wait = 100)) #num iterations to go past min test error before stop
dir_cat_d = catboost.train(learn_pool = cat_train_dir_d, test_pool = cat_test_dir_d, params = list(metric_period = 50,
                                                                                                   #iterations = 3000,
                                                                                                   od_type = "Iter", 
                                                                                                   od_wait = 50))

speed_cat_o = catboost.train(learn_pool = cat_train_speed_o, test_pool = cat_test_speed_o, params = list(metric_period = 50,
                                                                                                         #iterations = 1000,
                                                                                                         od_type = "Iter", 
                                                                                                         od_wait = 100))
speed_cat_d = catboost.train(learn_pool = cat_train_speed_d, test_pool = cat_test_speed_d, params = list(metric_period = 50,
                                                                                                         #iterations = 1000,
                                                                                                         od_type = "Iter", 
                                                                                                         od_wait = 100))
#calculate residual variance for log-bias correction
# o_speed_correction = var(log(test_o_labels$fut_s) - catboost.predict(model = speed_cat_o, pool = cat_test_speed_o))
# d_speed_correction = var(log(test_d_labels$fut_s) - catboost.predict(model = speed_cat_d, pool = cat_test_speed_d))
#seems to make predictions worse


acc_cat_o = catboost.train(learn_pool = cat_train_acc_o, test_pool = cat_test_acc_o, params = list(metric_period = 50,
                                                                                                   #iterations = 3000,
                                                                                                   od_type = "Iter", 
                                                                                                   od_wait = 100))
acc_cat_d = catboost.train(learn_pool = cat_train_acc_d, test_pool = cat_test_acc_d, params = list(metric_period = 50,
                                                                                                   #iterations = 3000,
                                                                                                   od_type = "Iter", 
                                                                                                   od_wait = 100))

#feature importance
catboost.get_feature_importance(dir_cat_o)
catboost.get_feature_importance(dir_cat_d)
catboost.get_feature_importance(speed_cat_o)
catboost.get_feature_importance(speed_cat_d)
catboost.get_feature_importance(acc_cat_o)
catboost.get_feature_importance(acc_cat_d)


#' experiment with:
#'  1. fit model with automatic test set evaluation thing
#'  2. shrink model
#'  3. drop unused features
#'  
#'  see if any of these improve performance


#df to generate predictions on
data_mod_test_pred = data_mod_test %>% 
  filter(throw == "post" | (throw == "pre" & lead(throw) == "post")) %>% #filter for post throw only
         #mutate the unknowns to be NA to ensure the model isn't using any future known data
         mutate(across(-c(game_player_play_id, game_play_id, throw, frame_id, play_direction, 
                   absolute_yardline_number, player_name, player_height, player_weight, player_position,
                   player_side, player_role, ball_land_x, ball_land_y, prop_play_complete), 
                ~ ifelse(throw == "pre", .x, NA))) 


#above should be CV
#then predict on entire data set



#do this in parallel
library(foreach)
library(doFuture)
library(progressr)
handlers(global = TRUE)
handlers("progress")

registerDoFuture()
plan(multisession, workers = 10)

#these are what we should parallelize over 
game_play_ids = data_mod_test_pred$game_play_id %>% unique()

#for testing
#game_play_ids = game_play_ids[1:10]

#this is my predict() function

#' flow of this:
#'  -loop through all the plays
#'    -loop through all the frames in a play
#'      -loop through the players in the play "post throw" (player_to_predict == TRUE)
#'        -predict next x,y
#'        -derive features necessary for next dir, s, a prediction
#'        -predict next dir, s, a

set.seed(1999)
start = Sys.time()
with_progress({
  p = progressor(steps = length(game_play_ids)) #progress
  
  results = foreach(group_id = game_play_ids, .combine = rbind, .packages = c("tidyverse", "doParallel", "catboost")) %dopar% {
    p(sprintf("Iteration %d", group_id)) #progress 
    
    #single player on single play
    curr_game_play_group = data_mod_test_pred %>% filter(game_play_id == group_id)
    #the frames in the play
    frames = curr_game_play_group$frame_id %>% unique()
    #player ids in the play we need to predict
    player_ids = curr_game_play_group$game_player_play_id %>% unique()
    
    #loop through frames in play (not in parallel)
    foreach(frame = frames, .combine = rbind) %do% {
      
      #info for all players in current frame
      curr_frame_all_players = data_mod_test_pred %>% filter(game_play_id == group_id, frame_id == frame)
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
        
        #closest player features
        closest_player_features = get_closest_player_min_dist_dir(curr_frame_all_players) 
        curr_frame_all_players = curr_frame_all_players %>% 
          select(-c(closest_player_dist)) %>% #deselect the closest player columns so we can merge the right ones
          full_join(closest_player_features, by = c("game_player_play_id")) %>% #join the min dist/dir
          mutate(closest_player_dir_diff = min_pos_neg_dir(est_dir - closest_player_dir)) %>%
          select(-closest_player_dir) #deselect the actual direction
        
        #all other features
        #need previous frame to compute lag stuff
        prev_curr_frame_df = curr_frame_all_players %>%
          rbind(prev_frame_all_players) %>%
          arrange(game_player_play_id, frame_id)
        
        #derived other features
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
      curr_frame_all_players = foreach(player = player_ids, .combine = rbind) %do% { #need to do this in a loop since catboost load pool thing
        cat_pred_df = curr_frame_all_players %>%
          filter(game_player_play_id == player) %>%
          select(all_of(rownames(dir_cat_d$feature_importances))) %>% #pred df for catboost
          catboost.load_pool()
        
        #return row with predicted dir, s, a
        curr_frame_player = curr_frame_all_players %>% 
          filter(game_player_play_id == player) %>% #predict on single player
          mutate(pred_dir = est_dir + ifelse(player_side == "Offense", 
                                             catboost.predict(dir_cat_o, cat_pred_df), 
                                             catboost.predict(dir_cat_d, cat_pred_df)),
                 pred_s = ifelse(player_side == "Offense", 
                                 catboost.predict(speed_cat_o, cat_pred_df), 
                                 catboost.predict(speed_cat_d, cat_pred_df)),
                 pred_a = est_acc + ifelse(player_side == "Offense", 
                                           catboost.predict(acc_cat_o, cat_pred_df), 
                                           catboost.predict(acc_cat_d, cat_pred_df)))
          #mutate(pred_s = ifelse(pred_s < 0, 0, pred_s)) #speed cannot be 0
        #this seems like a band aid fix - how to model speed as strictly positive? model on log scale then back-transform?
        
        #same colnames as previous iteration
        if(curr_frame_player$throw != "pre") {curr_frame_all_players = curr_frame_all_players %>% select(all_of(colnames(result)))} 
        
        curr_frame_player
      }
      
      ### return result ###
      result = curr_frame_all_players
      result
      
      #print(colnames(result))
      #if(frame %in% c(26,27)) {print(colnames(result))}
    }
  }
})
end = Sys.time()
end - start
#results

# I think this for loop is now memory intensive, rather than CPU intensive
# - if we can somehow make it less memory intesive that should speed this up


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
#' 
#' 


#the true x,y,dir,s,a values
true_vals = data_mod_test %>% 
  rename(true_dir = est_dir, true_s = est_speed, true_a = est_acc, true_x = x, true_y = y) %>% 
  select(game_player_play_id, frame_id, true_dir, true_s, true_a, true_x, true_y)

#bind results into df
results_pred = results %>%
  left_join(true_vals, by = c("game_player_play_id", "frame_id")) %>% #join true x,y values
  arrange(game_play_id, game_player_play_id, frame_id)


#evaluate dir, s, a results
results_pred %>% select(pred_dir, true_dir) %>% summary()
results_pred %>% select(pred_dir, true_dir) %>% mutate(pred_dir = pred_dir %% 360) %>% summary()
results_pred %>% select(pred_s, true_s) %>% summary()
results_pred %>% select(pred_a, true_a) %>% summary()

#I think we need to mod 360 the direction?
#also some predicted speeds are negative, clearly this is impossible - when this is the case, convert it to 0?
#is there a way for catboost to constrain the response?


#true vs predicted dir
ggplot(data = results_pred, mapping = aes(x = true_dir, y = min_pos_neg_dir(pred_dir - true_dir))) +
  geom_scattermore() +
  xlim(c(0, 360)) + 
  theme_bw() +
  labs(x = "True Direction", y = "Predicted Direction Minus True Direction")

#true vs predicted speed
ggplot(data = results_pred, mapping = aes(x = true_s, y = pred_s)) +
  geom_scattermore() +
  xlim(c(min(c(results_pred$pred_s, results_pred$true_s))), 
       max(c(results_pred$pred_s, results_pred$true_s))) + 
  ylim(c(min(c(results_pred$pred_s, results_pred$true_s))), 
       max(c(results_pred$pred_s, results_pred$true_s))) +
  theme_bw() +
  labs(x = "True Speed", y = "Predicted Speed")

#true vs predicted acc
ggplot(data = results_pred, mapping = aes(x = true_a, y = pred_a)) +
  geom_scattermore() +
  xlim(c(min(c(results_pred$pred_a, results_pred$true_a))), 
       max(c(results_pred$pred_a, results_pred$true_a))) + 
  ylim(c(min(c(results_pred$pred_a, results_pred$true_a))), 
       max(c(results_pred$pred_a, results_pred$true_a))) +
  theme_bw() +
  labs(x = "True Acceleration", y = "Predicted Acceleration")
#acceleration seems squashed a bit, maybe include higher true vals in training set?



#pred dir, s, a vs true dir, s, a
group_id = 1
dir_s_a_eval(group_id)

#single player movement
curr_game_player_play_id = results_pred %>% 
  group_by(game_player_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_player_play_id) %>% unique()
#curr_game_player_play_id = 1

plot_player_movement_pred(group_id = curr_game_player_play_id,
                          group_id_preds = results_pred %>% 
                            filter(game_player_play_id == curr_game_player_play_id) %>%
                            select(frame_id, x, y) %>%
                            rename(pred_x = x, pred_y = y))


#multiple players on play
group_id = 1
curr_game_play_id = results_pred %>% 
  group_by(game_play_id) %>%
  filter(cur_group_id() == group_id) %>% 
  pull(game_play_id) %>% unique()
#curr_game_play_id = 2156

multi_player_movement_pred(group_id = curr_game_play_id,
                           group_id_preds = results_pred %>%
                             filter(game_play_id == curr_game_play_id) %>%
                             select(game_player_play_id, frame_id, x, y) %>%
                             rename(pred_x = x, pred_y = y))

#lots of players dont have predictions since they weren't in the test set


### RMSE ###

#worst player_plays
results_rmse_player = results_pred %>%
  filter(throw == "post") %>%
  group_by(game_player_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
results_rmse_player %>% arrange(desc(rmse))

#worst plays
results_rmse_play = results_pred %>%
  filter(throw == "post") %>%
  group_by(game_play_id) %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
results_rmse_play %>% arrange(desc(rmse))


#plot player rmse
results_rmse_player %>% 
  ggplot(mapping = aes(y = rmse)) +
  geom_boxplot() +
  theme_bw()

#plot play rmse
results_rmse_play %>% 
  ggplot(mapping = aes(y = rmse)) +
  geom_boxplot() +
  theme_bw()



#across entire dataset
results_pred %>% 
  filter(throw == "post") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))
#remember this is only 20% of the dataset


#fit on all -                     - 1.111


#fit on prop_play_complete > 0.4  - 1.083

#fit on prop_play_complete > 0.4  - 1.035
#    and separate off,def models 


#fit on throw == "post"           - 1.154


#cat boost off/def models and 0.4 prop_play  - 0.951

#Fixed acceleration!
#catboost off/def models, 0.4 prop_play -  0.834
#above but added closest player features - 0.821

#same as above but filtering out weird data - 0.730
#same as above but band-aid fix to negative speeds - 0.729


#0.7609 - no log-bias speed correction



#offense across entire dataset
results_pred %>% 
  filter(player_side == "Offense", throw == "post") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))

#defense across entire dataset
results_pred %>% 
  filter(player_side == "Defense", throw == "post") %>%
  summarise(rmse = get_rmse(true_x = true_x, true_y = true_y,
                            pred_x = x, pred_y = y))

#defense is much harder to predict 

#' Off - 0.537
#' Def - 0.929
#' 
#' 
#' 
#'  YOU NEED TO DO CV - ALL THESE RMSEs MIGHT BE USING DIFFERENT TEST SETS



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
#' 
#' 8. Create different models for offense and defense?
#'       defence needs different features, ie, dist/dir to offensive player, or dist/dir to other player...



