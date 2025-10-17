#various helper functions

library(tidyverse)
library(here)



#' function to calculate direction between two frames
#' direction in degrees and on the coordinate system used by BDB (0 deg is vertical, clockwise)
#' inputs: x_diff, y_diff between current and previous frame

get_dir = function(x_diff, y_diff) {
  (90 - (atan2(y = y_diff, x = x_diff)*180/pi)) %% 360 
}
#be careful when there's no change in position (x_diff = y_diff = 0) - returns 90


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

