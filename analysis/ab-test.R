library(tidyverse)
library(lubridate)
df_ratings <- read_csv("data/new_df_full.csv")

#rating per user

df_ratings %>% 
  mutate(date = as_datetime(time)) %>% 
  mutate(new_design = (year(date) == 2022)) %>% 
  #mutate(new_design = (date > now() - 5 * 3600)) %>% 
  group_by(userID, new_design) %>% 
  summarise(count = n()) %>% 
  ggplot() +
  geom_histogram(aes(x = count, fill=new_design))

df_ratings %>% 
  mutate(date = as_datetime(time)) %>% 
  mutate(new_design = (year(date) == 2022)) %>% 
  group_by(new_design, userID) %>% 
  summarise(count = n()) %>% 
  summarise(m = mean(count), s = sd(count))
  

#convertion rate

df_positions <- read_csv("data/positions.csv") %>% 
  filter(userID != "16421609")
  

df_positions %>% 
  mutate(date = as_datetime(time)) %>% 
  ggplot() +
  geom_histogram(aes(x = date))
  

df_positions %>% 
  group_by(session_id) %>% 
  summarise(duration = max(time) - min(time)) %>% 
  filter(duration < 1000, duration > 10) %>% 
  ggplot() +
  geom_histogram(aes(duration))

df_positions %>% 
  group_by(session_id) %>% 
  summarise(duration = max(time) - min(time)) %>% 
  filter(duration < 1000, duration > 10) %>% 
  ggplot() +
  geom_histogram(aes(duration))

df_positions %>% 
  group_by(session_id) %>% 
  summarise(duration = max(time) - min(time)) %>% 
  mutate(short = if_else(duration < 30, T, F)) %>% 
  group_by(short) %>% 
  summarise(count = n())


df_positions %>% 
  filter(position == "results") %>% 
  select(userID) %>% 
  distinct()

df_positions %>% 
  group_by(session_id, position) %>% 
  summarise(count = n())
  distinct()
  

df_positions %>% 
  tidyr::expand(session_id, position) %>% 
  left_join(df_positions %>% 
              group_by(session_id, position) %>% 
              summarise(count = n())) %>% 
  group_by(position) %>% 
  summarise(proba = sum(!is.na(count)))
  
