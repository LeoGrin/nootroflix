library(tidyverse)
library(ggrepel)

theme_set(theme(plot.background = element_rect(fill="#fffff8"), #to incorporte into the main article
                text = element_text(size = 16)))

df <- read_csv("data/nootroflix_ssc_ratings_clean.csv")

df %>% 
  select(userID, itemID, rating, issue) %>% 
  write_csv("data/data_with_ssc.csv")

df %>% filter(itemID == "Cerebrolysin")

df_results <- read_csv("analysis/analysis_results/results_summary.csv")# %>% 
  #mutate(nootropic = str_replace(nootropic, "- ", ", ")) %>% 
  #mutate(nootropic = if_else(nootropic == "7-8-dihydroxyflavone", "7,8-dihydroxyflavone", nootropic))


# df_results %>%
#   left_join(
#     df %>%
#     mutate(nootropic = itemID) %>%
#     group_by(nootropic) %>%
#     summarise(count = n()) %>%
#     select(nootropic, count),
#     by = c("nootropic")) %>%
#   relocate(nootropic, count) %>% 
#   write_csv("analysis/analysis_results/results_summary.csv")

View(df_results)


df %>% 
  filter(itemID == "7,8-dihydroxyflavone")

df_results %>% 
  mutate(nootropic = if_else(nootropic == "7-8-dihydroxyflavone", "7,8-dihydroxyflavone", nootropic))

df %>% 
  mutate(nootropic = itemID) %>% 
  mutate(nootropic = if_else(nootropic == "7,8-dihydroxyflavone", "7-8-dihydroxyflavone", nootropic)) %>% 
  group_by(nootropic) %>% 
  summarise(count = n()) %>% 
  left_join(df_results, on = "nootropic")

View(df %>% 
  filter(itemID == "Zembrin" | itemID == "Kanna (except Zembrin)"))


df %>% 
  mutate(nootropic = itemID) %>% 
  group_by(nootropic) %>% 
  summarise(count = n()) %>% 
  left_join(df_results, on = "nootropic") %>% 
  ggplot() +
  ggrepel::geom_text_repel(aes(x = count, y = estimated_mean_rating, label=nootropic), segment.colour = NA)# +
#scale_x_continuous(trans='log2')


df %>% 
  filter(itemID == "Psilocybin (microdose)") %>% 
  ggplot() + 
  geom_histogram(aes(x = rating))

df %>% 
  filter(itemID == "LSD (microdose)") %>% 
  ggplot() + 
  geom_histogram(aes(x = rating))

df %>% 
  filter(itemID == "Adderall") %>% 
  ggplot() + 
  geom_histogram(aes(x = rating))

## Users who have used zembrin and kanna

df %>% 
  filter(itemID == "Zembrin" | itemID == "Kanna (except Zembrin)") %>% 
  group_by(userID) %>% 
  filter(n() >=2) %>% 
  group_by(itemID) %>% 
  summarise(a = mean(rating), b = sd(rating))


View(df_results)

df_metadata <- read_csv2("data/nootropics_metadata.csv")




#### PLOTS

df_results %>% 
  filter(str_detect(nootropic, "diet") | str_detect(nootropic, "fasting")) %>%
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, estimated_mean_rating)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = estimated_mean_rating, xmax = estimated_mean_rating.upper, xmin = estimated_mean_rating.lower, y = nootropic)) +
  xlab("Estimated mean rating") + 
  ylab("")

ggsave("analysis/plots/diets_mean_ratings.jpeg", width=10, height=4, units = "in", limitsize = F, dpi=300)


df_results %>% 
  filter(str_detect(nootropic, "diet") | str_detect(nootropic, "fasting")) %>%
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, proba_life_changing)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = proba_life_changing, xmax = proba_life_changing.upper, xmin = proba_life_changing.lower, y = nootropic)) +
  xlab("Probability of being life changing") + 
  ylab("")


ggsave("analysis/plots/diets_life_changing.jpeg", width=10, height=4, units = "in", limitsize = F, dpi=300)


df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  filter(type == "Lifestyle" | nootropic == "Modafinil" | nootropic == "Ginseng" ) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, proba_life_changing)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = proba_life_changing, xmax = proba_life_changing.upper, xmin = proba_life_changing.lower, y = nootropic, color=type=="Lifestyle")) + 
  theme(legend.position="none") +
  xlab("Probability of being life changing") + 
  ylab("")
  

ggsave("analysis/plots/lifestyle_life_changing.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)


#Peptides
df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  filter(type == "Peptides"  | nootropic == "Modafinil" | nootropic == "Ginseng" ) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, estimated_mean_rating)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = estimated_mean_rating, xmax = estimated_mean_rating.upper, xmin = estimated_mean_rating.lower, y = nootropic, color=type=="Peptides")) + 
  theme(legend.position="none") +
  xlab("Estimated mean rating") + 
  ylab("")

ggsave("analysis/plots/ratings_mean_peptides.jpeg", width=10, height=5, units = "in", limitsize = F, dpi=300)


df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  filter(type == "Peptides"  | nootropic == "Modafinil" | nootropic == "Ginseng" ) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, proba_life_changing)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = proba_life_changing, xmax = proba_life_changing.upper, xmin = proba_life_changing.lower, y = nootropic, color=type=="Peptides")) + 
  theme(legend.position="none") +
  xlab("Probability of changing your life") + 
  ylab("")

ggsave("analysis/plots/ratings_life_changing_peptides.jpeg", width=10, height=5, units = "in", limitsize = F, dpi=300)

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  filter(type == "Peptides"  | nootropic == "Modafinil" | nootropic == "Ginseng" ) %>% 
  mutate(nootropic = if_else(str_detect(nootropic, "Semax"), "Semax", nootropic)) %>%
  mutate(nootropic = if_else(str_detect(nootropic, "Selank"), "Selank", nootropic)) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, count)) %>% 
  ggplot() +
  geom_bar(aes(x = nootropic, y=count, fill=type=="Peptides"), stat = "identity") + #%>% 
  theme(legend.position="none") +
  xlab("") + 
  ylab("Number of ratings")

ggsave("analysis/plots/count_peptides.jpeg", width=10, height=5, units = "in", limitsize = F, dpi=300)

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Peptides"  |nootropic == "Modafinil" | nootropic == "Ginseng" ) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, estimated_mean_rating)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = estimated_mean_rating, xmax = estimated_mean_rating.upper, xmin = estimated_mean_rating.lower, y = nootropic, color=type=="Peptides")) + 
  theme(legend.position="none") +
  xlab("Estimated mean rating") + 
  ylab("")

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
 # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Herbs"  |nootropic == "Modafinil" | nootropic == "Ginseng" ) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, estimated_mean_rating)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = estimated_mean_rating, xmax = estimated_mean_rating.upper, xmin = estimated_mean_rating.lower, y = nootropic, color=type=="Peptides")) + 
  theme(legend.position="none") +
  xlab("Estimated mean rating") + 
  ylab("")

ggsave("analysis/plots/lifestyle_mean_ratings.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)


df_issues <- read_csv("analysis/analysis_results/issues_summary.csv")

df_issues %>% 
  filter(str_detect(nootropic, "diet") | str_detect(nootropic, "fasting")) %>% 
  filter(variant == "stan model") %>% 
  filter(issue %in% c("side_effects", "long_term_side_effects")) %>% 
  mutate(nootropic = fct_reorder(nootropic, prop)) %>% 
  ggplot() +
  geom_pointrange(aes(x = prop, xmax = prop_high, xmin=prop_low, y=nootropic, color=issue), position = position_dodge2(width=0.2))+
  xlab("Probability of issue") + 
  ylab("")

ggsave("analysis/plots/issues_diets.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)


# Peptides

df %>%




