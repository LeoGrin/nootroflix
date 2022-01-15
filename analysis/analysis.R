library(tidyverse)

df_ratings <- read_csv("data/new_df.csv")
df_users <- read_csv("data/users.csv") %>% 
  mutate(focus = if_else(is.na(focus), for_focus, focus),
         cognition = if_else(is.na(cognition), for_cognition, cognition),
         mood = if_else(is.na(mood), for_mood, mood),
         anxiety = if_else(is.na(anxiety), for_anxiety, anxiety),
         motivation = if_else(is.na(motivation), for_motivation, motivation)) %>% 
  select(-for_focus, -for_anxiety, -for_motivation, -for_cognition, -for_mood)

df_metadata <- read_delim("data/nootropics_metadata.csv", delim=";")

#df_users %>% write_csv("data/users_fixed.csv")

df_ratings %>%
  left_join(df_users %>% mutate(userID = as.numeric(userID)), by = c("userID")) %>% 
  filter(libido == "Yes, a major reason") %>% 
  group_by(itemID) %>% 
  summarise(mean_rating = mean(rating), sd_rating = sd(rating), count = n()) %>% 
  arrange(mean_rating)



df <-  df_ratings %>%
  left_join(df_users %>% mutate(userID = as.numeric(userID)), by = c("userID")) 

df_users %>% 
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


l <- lm(rating ~ gender + age + anxiety + mood + motivation + focus, data = df %>% filter(itemID == "Modafinil"))

summary(l)

library(lme4)

l <- lmer(rating ~ (1 | userID) + gender + age + anxiety + mood + motivation + focus, data = df)

l <- lmer(rating ~ (1 | userID) + (1 + anxiety | itemID), data = df)

library(rstanarm)
options(mc.cores = parallel::detectCores())

df <- df %>% 
  mutate(
    motivation = case_when(
    motivation == "Yes, a major reason" ~ 3,
    motivation == "Yes, a minor reason" ~ 1,
    motivation == "Not at all a reason" ~ 0),
    focus = case_when(
      focus == "Yes, a major reason" ~ 3,
      focus == "Yes, a minor reason" ~ 1,
      focus == "Not at all a reason" ~ 0),
    mood = case_when(
      mood == "Yes, a major reason" ~ 3,
      mood == "Yes, a minor reason" ~ 1,
      mood == "Not at all a reason" ~ 0),
    anxiety = case_when(
      anxiety == "Yes, a major reason" ~ 3,
      anxiety == "Yes, a minor reason" ~ 1,
      anxiety == "Not at all a reason" ~ 0),
    cognition = case_when(
      cognition == "Yes, a major reason" ~ 3,
      cognition == "Yes, a minor reason" ~ 1,
      cognition == "Not at all a reason" ~ 0),
    libido = case_when(
      libido == "Yes, a major reason" ~ 3,
      libido == "Yes, a minor reason" ~ 1,
      libido == "Not at all a reason" ~ 0)
    )

l <- stan_glmer(rating ~ (1 + anxiety + focus + motivation + libido + cognition + mood | itemID) + (1 | userID), data = df,
                family = gaussian(link = "identity"))

l_ordered_factor <- stan_glmer(rating ~ (1 + anxiety + focus + motivation + libido + cognition + mood | itemID) + (1 | userID), 
                               data = df %>% 
                                 mutate(motivation = as_factor(motivation),
                                        focus = as_factor(focus),
                                        mood = as_factor(mood),
                                        anxiety = as_factor(anxiety),
                                        cognition = as_factor(cognition),
                                        libido = as_factor(libido)),
                               family = gaussian(link = "identity"))


(coef(l)$itemID)$anxiety


library(tidybayes)

get_variables(l_ordered_factor)

to_plot_ordered_factor <- l_ordered_factor %>%
  spread_draws(b[term,group]) %>% 
  filter(term %in% c("anxiety1", "anxiety3")) %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  group_by(group, .chain, .iteration, .draw) %>% 
  pivot_wider(names_from = term, values_from=b) %>% 
  mutate(b = (anxiety1 - anxiety3) / 2) %>% 
  select(-anxiety1, -anxiety3) %>% 
  mutate(term = "anxiety") %>% 
  median_qi(condition_mean = b, .width = c(.95, .66)) %>%
  mutate(group = fct_reorder(group, condition_mean))

to_plot <- l %>%
  spread_draws(b[term,group]) %>% 
  filter(term == "anxiety") %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  median_qi(condition_mean = b, .width = c(.95, .66)) %>%
  mutate(group = fct_reorder(group, condition_mean))



l %>%
  spread_draws(b[term,group]) %>% 
  filter(term == "anxiety") %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  median_qi(condition_mean = b, .width = c(.3)) %>% 
  mutate(group = fct_reorder(group, condition_mean)) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()
  #filter(.lower >= 0)

View(l %>%
  spread_draws(b[term,group]) %>% 
  filter(term == "focus") %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  median_qi(condition_mean = b, .width = c(.3)) %>% 
  filter(.lower >=0))

l %>%
       spread_draws(b[term,group]) %>% 
       filter(term == "libido") %>% 
       mutate(group = str_remove(group, "itemID:")) %>% 
       median_qi(condition_mean = b, .width = c(.4)) %>% 
       filter(.lower >=0) %>% 
  arrange(-condition_mean)

l %>%
       spread_draws(b[term,group]) %>% 
       filter(term == "motivation") %>% 
       mutate(group = str_remove(group, "itemID:")) %>% 
       median_qi(condition_mean = b, .width = c(.4)) %>% 
       filter(.upper <=0) %>% 
  arrange(condition_mean)

l %>%
  spread_draws(b[term,group]) %>% 
  filter(term == "focus") %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  median_qi(condition_mean = b, .width = c(.66, 0.9)) %>% 
  filter(str_detect(group, "italin"))


#best

to_plot %>% 
  filter(condition_mean > 0) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

to_plot_ordered_factor %>% 
  filter(condition_mean > 0) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()


summary(l)



l <- lmer(rating ~ (1 | itemID) + (1 | userID), data = df)

#is rating > 0

l_effective <- stan_glmer(is_effective ~ (1 | itemID) + (1 | userID), data =  df %>% 
                            group_by(userID) %>% 
                            mutate(n_ratings = n(), min_rating = min(rating)) %>% 
                            filter(n_ratings > 10) %>%  #?
                            mutate(is_effective = if_else(rating > min_rating, 1, 0)),
                            family = binomial(link = "logit"))
df %>% 
  group_by(userID) %>% 
  mutate(n_ratings = n(), min_rating = min(rating)) %>% 
  filter(n_ratings > 10) %>%  #?
  mutate(is_effective = if_else(rating > min_rating, 1, 0)) #%>%  #adjust for the fact that a lot of people haven't understood that 1 = positive
  #group_by(itemID) %>% 
  #summarise(a = mean(is_effective))

df %>% 
  group_by(userID) %>% 
  summarise(n_ratings = n(), min_rating = min(rating)) %>% 
  filter(n_ratings > 10) %>% 
  group_by(min_rating) %>% 
  summarise(count = n()) %>% 
  mutate(prop = 100 * count / sum(count))


df %>% 
  mutate(life_changing = if_else(rating == 10, 1, 0)) %>%  #adjust for the fact that a lot of people haven't understood that 1 = positive
  group_by(userID) %>% 
  summarise(n_life_changing = sum(life_changing)) %>% 
  group_by(n_life_changing) %>% 
  summarise(count = n())

df %>% 
  mutate(life_changing = if_else(rating == 10, 1, 0)) %>%  #adjust for the fact that a lot of people haven't understood that 1 = positive
  summarise(proba_life_changing = mean(life_changing), count = n())

View(df %>% 
  mutate(life_changing = if_else(rating == 10, 1, 0)) %>%  #adjust for the fact that a lot of people haven't understood that 1 = positive
  group_by(userID) %>% 
  mutate(n_life_changing = sum(life_changing), count = n()) %>% 
  filter(n_life_changing / count < 0.2) %>% 
  ungroup() %>% 
  group_by(itemID) %>% 
  summarise(proba_life_changing = mean(life_changing), count = n()) %>% 
  arrange(proba_life_changing))



library(rstanarm)
options(mc.cores = parallel::detectCores())

l <- stan_glmer(rating ~ (1 | itemID) + (1 | userID), data = df,
                  family = gaussian(link = "identity"),
                  seed = 12345)


library(tidybayes)

to_plot <- l %>%
  spread_draws(`(Intercept)`, b[,group])%>%
  filter(str_detect(group, "itemID:")) %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  median_qi(condition_mean = `(Intercept)` + b, .width = c(.95, .66)) %>%
  mutate(group = fct_reorder(group, condition_mean))


to_plot_effective <- l_effective %>%
  spread_draws(`(Intercept)`, b[,group])%>%
  filter(str_detect(group, "itemID:")) %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  median_qi(condition_mean = `(Intercept)` + b, .width = c(.95, .66)) %>%
  mutate(group = fct_reorder(group, condition_mean))


#best
to_plot %>% 
  filter(group %in% (to_plot %>% filter(.width==0.95) %>% filter(.upper > 4.5))$group) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

to_plot_effective %>% 
  filter(group %in% (to_plot %>% filter(.width==0.95) %>% filter(.upper > 4.5))$group) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

to_plot %>%
  mutate(proba = F) %>% 
  bind_rows(to_plot_effective %>% mutate(proba = T)) %>% 
  filter(group %in% (to_plot %>% filter(.width==0.95) %>% filter(.upper > 4.5))$group) %>% 
  ggplot(aes(y = group, x = condition_mean, color=proba, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()


#worse
to_plot %>% 
  filter(group %in% (to_plot %>% filter(.width==0.95) %>% filter(.upper < 4.5))$group) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

to_plot_effective %>% 
  filter(group %in% (to_plot %>% filter(.width==0.95) %>% filter(.upper < 4.5))$group) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

#most used
to_plot %>% 
  left_join(df %>% 
             mutate(itemID = str_replace_all(itemID, " ", "_")) %>% 
             group_by(itemID) %>% 
             summarise(count = n()) %>% 
             arrange(-count), by = c("group"="itemID")) %>%
  filter(rank(-count) < 40) %>% 
  mutate(group = fct_reorder(group, condition_mean)) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper, size=count)) +
  geom_pointinterval() +
  geom_point()

to_plot_effective %>% 
  left_join(df %>% 
              mutate(itemID = str_replace_all(itemID, " ", "_")) %>% 
              group_by(itemID) %>% 
              summarise(count = n()) %>% 
              arrange(-count), by = c("group"="itemID")) %>%
  filter(rank(-count) < 40) %>% 
  mutate(group = fct_reorder(group, condition_mean)) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper, size=count)) +
  geom_pointinterval() +
  geom_point()

library(ggrepel)

to_plot %>% 
  select(group, condition_mean) %>% 
  distinct() %>% 
  left_join(df %>% 
              mutate(itemID = str_replace_all(itemID, " ", "_")) %>% 
              group_by(itemID) %>% 
              summarise(count = n()), by = c("group" = "itemID")) %>% 
  ggplot() +
  ggrepel::geom_text_repel(aes(x = count, y = condition_mean, label=group), segment.colour = NA)# +
  #scale_x_continuous(trans='log2')


  #geom_point(aes(x = count, y = condition_mean))+




#by category
to_plot %>% 
  filter(group %in% (df_metadata %>% 
                       filter(str_detect(type, "Lifestyle")) %>% 
                       mutate(nootropic_short = str_replace_all(nootropic_short, " ", "_")))$nootropic_short) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

to_plot %>% 
  filter(group %in% (df_metadata %>% 
                       filter(str_detect(type, "Modafinil")) %>% 
                       mutate(nootropic_short = str_replace_all(nootropic_short, " ", "_")))$nootropic_short) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

to_plot_effective %>% 
  filter(group %in% (df_metadata %>% 
                       filter(str_detect(type, "Modafinil")) %>% 
                       mutate(nootropic_short = str_replace_all(nootropic_short, " ", "_")))$nootropic_short) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

to_plot %>% 
  filter(group %in% (df_metadata %>% 
                       filter(str_detect(type, "Racetams")) %>% 
                       mutate(nootropic_short = str_replace_all(nootropic_short, " ", "_")))$nootropic_short) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

#To try:
# not mean, but median, or proportion over 0 (or 1?)



#plotting distribution

l %>%
  spread_draws(`(Intercept)`, b[,group])%>%
  filter(str_detect(group, "itemID:")) %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  filter(group %in% (df_metadata %>% 
                       filter(str_detect(type, "Modafinil")) %>% 
                       mutate(nootropic_short = str_replace_all(nootropic_short, " ", "_")))$nootropic_short) %>% 
  mutate(condition_mean = `(Intercept)` + b) %>%
  mutate(group = fct_reorder(group, condition_mean)) %>% 
  ggplot(aes(x = condition_mean, fill = group)) +
  stat_halfeye(alpha=0.5)

#######
l %>%
  spread_draws(`(Intercept)`, b[,group])%>%
  filter(str_detect(group, "itemID:")) %>% 
  mutate(group = str_remove(group, "itemID:")) %>% 
  median_qi(condition_mean = `(Intercept)` + b, .width = c(.95, .66)) %>%
  mutate(group = fct_reorder(group, condition_mean)) %>% 
  filter(condition_mean <= 5, condition_mean > 4) %>% 
  ggplot(aes(y = group, x = condition_mean, xmin = .lower, xmax = .upper)) +
  geom_pointinterval()

l %>%
  spread_draws(b[term,group]) %>%
  head(10)
  ggplot(aes(y = fct_rev(condition), x = condition_mean)) +
  stat_pointinterval()

x <- as.matrix(l, regex_pars = "\\(Intercept\\) itemID")

bayesplot::mcmc_intervals(x)


my_labels <- (df %>% select(itemID) %>% arrange(itemID) %>% distinct())$itemID
plot(l, regex_pars = "\\(Intercept\\) itemID")+ 
  ggplot2::scale_y_discrete(labels = my_labels)

bayesplot::mcmc_areas_ridges(l, regex_pars = "itemID:(*)")


df_res <- coef(l)$itemID
df_res <- cbind(nootropic = rownames(df_res), df_res)
rownames(df_res) <- 1:nrow(df_res)
df_res <- tibble(df_res)
colnames(df_res) <- c("nootropic", "rating")




df_res %>% 
  mutate(nootropic = fct_reorder(nootropic, rating)) %>% 
  ggplot() +
  geom_point(aes(x = rating, y=nootropic))

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
