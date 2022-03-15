library(tidyverse)

df_nootroflix_ratings <- read_csv("data/nootroflix_ratings.csv")
df_nootroflix_users <- read_csv("data/nootroflix_users.csv")
df_ssc <- read_csv("data/ssc_ratings.csv")

df_nootroflix_ratings <- df_nootroflix_ratings %>% 
  filter(is_true_ratings) %>% 
  group_by(userID) %>% 
  mutate(min_time = min(time)) %>% #"time" is the time when the user clicked "Get results"
  filter(time == min_time)  #for each user, we only keep the first use of the website
  
#View(df_nootroflix_users %>% 
#  select(permanent_favorite_noot) %>% 
#  distinct())

df_nootroflix_ratings %>% 
  write_csv("data/nootroflix_ratings_clean.csv")

df_nootroflix_ratings %>% 
  bind_rows(df_ssc) %>% 
  write_csv("data/nootroflix_ssc_ratings_clean.csv")

df_nootroflix_ratings %>% 
  right_join(df_nootroflix_users, by=c("userID", "time")) %>% 
  write_csv("data/nootroflix_ratings_users_clean.csv")
  
