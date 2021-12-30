library(tidyverse)

df_ratings <- read_csv("data/new_df.csv")
df_users <- read_csv("data/users.csv")

View(df_ratings %>%
  left_join(df_users %>% mutate(userID = as.numeric(userID)), by = c("userID")) %>% 
  filter(libido == "Yes, a major reason") %>% 
  group_by(itemID) %>% 
  summarise(mean_rating = mean(rating), sd_rating = sd(rating), count = n()) %>% 
  arrange(mean_rating))


df %>% 
  group_by(anxiety, mood) %>% 
  summarise(count = n())

df %>% 
  group_by(motivation) %>% 
  summarise(count = n())

df %>% 
  group_by(cognition) %>% 
  summarise(count = n())

df %>% 
  group_by(mood) %>% 
  summarise(count = n())

df %>% 
  group_by(focus) %>% 
  summarise(count = n())


l <- lm(rating ~ gender + age + anxiety + mood + motivation + focus, data = df_ratings %>%
          left_join(df_users %>% mutate(userID = as.numeric(userID)), by = c("userID")) %>% filter(itemID == "Ginseng"))

summary(l)

df_new <- read_csv("data/new_df.csv")
df_ssc <- read_csv("data/dataset_clean.csv")
View(df_new %>% 
  group_by(itemID) %>% 
  summarise(count = n()) %>% 
  arrange(count))

View(df_ssc %>% 
       group_by(itemID) %>% 
       summarise(count = n()) %>% 
       arrange(count))
