library(tidyverse)
library(ggrepel)

theme_set(theme(plot.background = element_rect(fill="#fffff8"), #to incorporte into the main article
                text = element_text(size = 16)))

df <- read_csv("data/nootroflix_ssc_ratings_clean.csv")

df %>% 
  filter(userID > 10000) %>% 
  left_join(df_metadata %>% 
              select(nootropic, type), by=c("itemID" = "nootropic")) %>% 
  filter(type == "Anti-depressants") %>% 
  group_by(userID) %>% 
  mutate(count = n()) %>% 
  filter(itemID == "Bupropion (Wellbutrin, Zyban...)") %>% 
  select(itemID, userID, count) %>% 
  ungroup() %>% 
  summarise(median = median(count), mean=mean(count))


df_issues <- read_csv("analysis/analysis_results/issues_summary.csv")

df_issues %>% 
  left_join(df_metadata %>% 
              select(nootropic, type), on="nootropic") %>% 
  filter(str_detect("Peptides", type) | nootropic == "Modafinil" | nootropic == "Ginseng") %>% 
  filter(variant=="stan model") %>% 
  ggplot() +
  geom_pointrange(aes(y=nootropic, x = prop, xmax=prop_high, xmin=prop_low, color=issue), position = position_dodge2(width=0.5)) +
  ylab("")+
  xlab("Probability of issue")

ggsave("analysis/plots/issues_peptides.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)
df %>%

df %>% 
  select(userID, itemID, rating, issue) %>% 
  write_csv("data/data_with_ssc.csv")

df %>% filter(itemID == "Cerebrolysin")

df_results <- read_csv("analysis/analysis_results/results_summary.csv")# %>% 
  #mutate(nootropic = str_replace(nootropic, "- ", ", ")) %>% 
  #mutate(nootropic = if_else(nootropic == "7-8-dihydroxyflavone", "7,8-dihydroxyflavone", nootropic))

df_results %>% 
  filter(count_without_ssc > 10) %>% 
  #filter(rank(-count) < 70) %>% 
  ggplot() +
  #scale_x_reverse() + 
  ggrepel::geom_text_repel(aes(x = count, y = estimated_mean_rating, label=nootropic)) +
  annotate("rect", xmin = 300, xmax = +Inf,  ymin = -Inf, ymax = 4.5,   fill = "red", alpha=0.1) +
  annotate("rect", xmin = 0, xmax = 300,  ymin = 4.5, ymax = +Inf,   fill = "green", alpha=0.1) + 
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Estimated mean rating")

ggsave("analysis/plots/ggrepel_mean_ratings.jpeg", width=13, height=10, units = "in", limitsize = F, dpi=300)


df_results %>% 
  filter(count_without_ssc > 10) %>% 
  #filter(rank(-count) < 70) %>% 
  ggplot() +
  #scale_x_reverse() + 
  ggrepel::geom_text_repel(aes(x = count, y = proba_life_changing, label=nootropic)) +
  annotate("rect", xmin = 300, xmax = +Inf,  ymin = 0, ymax = 0.02,   fill = "red", alpha=0.1) +
  annotate("rect", xmin = 0, xmax = 300,  ymin = 0.02, ymax = +Inf,   fill = "green", alpha=0.1) + 
  scale_x_log10() + 
  scale_y_log10() +
  xlab("Number of ratings") +
  ylab("Probablity of being life changing")

ggsave("analysis/plots/ggrepel_life_changing_ratings.jpeg", width=13, height=10, units = "in", limitsize = F, dpi=300)



df %>% 
  filter(str_detect(itemID, "LSD")) %>% 
  ggplot() +
  geom_histogram(aes(x = rating)) +
  facet_wrap(~itemID)

ggsave("analysis/plots/l_micro_rating_distrib.jpeg", width=6, height=4, units = "in", limitsize = F, dpi=300)

df %>% 
  filter(str_detect(itemID, "Psilocybin")) %>% 
  ggplot() +
  geom_histogram(aes(x = rating)) +
  facet_wrap(~itemID)

ggsave("analysis/plots/p_micro_rating_distrib.jpeg", width=6, height=4, units = "in", limitsize = F, dpi=300)


# df_results %>%
#   left_join(
#     df %>%
#     mutate(nootropic = itemID) %>%
#     group_by(nootropic) %>%
#     summarise(count_without_ssc = n()) %>%
#     select(nootropic, count_without_ssc),
#     by = c("nootropic")) %>%
#   relocate(nootropic, count, count_without_ssc) %>%
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


# Stimulant

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Dopamine Modulators"  |nootropic == "Modafinil") %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, estimated_mean_rating)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = estimated_mean_rating, xmax = estimated_mean_rating.upper, xmin = estimated_mean_rating.lower, y = nootropic)) + 
  theme(legend.position="none") +
  xlab("Estimated mean rating") + 
  ylab("")

ggsave("analysis/plots/stimulant_mean_ratings.jpeg", width=10, height=5, units = "in", limitsize = F, dpi=300)

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Dopamine Modulators"  |nootropic == "Modafinil") %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, proba_life_changing)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = proba_life_changing, xmax = proba_life_changing.upper, xmin = proba_life_changing.lower, y = nootropic)) + 
  theme(legend.position="none") +
  xlab("Probability of being life-changing") + 
  ylab("")

ggsave("analysis/plots/stimulant_life_changing.jpeg", width=10, height=5, units = "in", limitsize = F, dpi=300)



df_issues %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Dopamine Modulators"  |nootropic == "Modafinil") %>% 
  filter(variant == "stan model") %>% 
  #filter(issue %in% c("side_effects", "long_term_side_effects")) %>% 
  #mutate(nootropic = fct_reorder(nootropic, prop)) %>% 
  ggplot() +
  geom_pointrange(aes(x = prop, xmax = prop_high, xmin=prop_low, y=nootropic, color=issue), position = position_dodge2(width=0.2))+
  xlab("Probability of issue") + 
  ylab("")

ggsave("analysis/plots/stimulant_issues.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)


# Racetams

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Racetams"  |nootropic == "Modafinil") %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, estimated_mean_rating)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = estimated_mean_rating, xmax = estimated_mean_rating.upper, xmin = estimated_mean_rating.lower, y = nootropic, color = type=="Racetams")) + 
  theme(legend.position="none") +
  xlab("Estimated mean rating") + 
  ylab("")

ggsave("analysis/plots/racetams_mean_ratings.jpeg", width=10, height=5, units = "in", limitsize = F, dpi=300)

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Racetams"  |nootropic == "Modafinil") %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, proba_life_changing)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = proba_life_changing, xmax = proba_life_changing.upper, xmin = proba_life_changing.lower, y = nootropic, color = type=="Racetams")) + 
  theme(legend.position="none") +
  xlab("Probability of being life-changing") + 
  ylab("")

ggsave("analysis/plots/racetams_life_changing.jpeg", width=10, height=5, units = "in", limitsize = F, dpi=300)



df_issues %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Racetams") %>% 
  filter(variant == "stan model") %>% 
  #filter(issue %in% c("side_effects", "long_term_side_effects")) %>% 
  #mutate(nootropic = fct_reorder(nootropic, prop)) %>% 
  ggplot() +
  geom_pointrange(aes(x = prop, xmax = prop_high, xmin=prop_low, y=nootropic, color=issue), position = position_dodge2(width=0.3))+
  xlab("Probability of issue") + 
  ylab("")

ggsave("analysis/plots/racetams_issues.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)


# Anti depressant

anti_depressant_list <- c("Bright lights in morning / Dawn simulator", "SAM-e", "Zembrin", "Polygala tenuifolia",
                          "St John's Wort", "Omega-3 Supplements", "Tryptophan")

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Anti-depressants" | nootropic %in% anti_depressant_list) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, estimated_mean_rating)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = estimated_mean_rating, xmax = estimated_mean_rating.upper, xmin = estimated_mean_rating.lower, y = nootropic)) + 
  theme(legend.position="none") +
  xlab("Estimated mean rating") + 
  ylab("")

ggsave("analysis/plots/antidepressant_mean_ratings.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)

df_results %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Anti-depressants" | nootropic %in% anti_depressant_list) %>% 
  mutate(nootropic = as_factor(nootropic)) %>% 
  mutate(nootropic =  fct_reorder(nootropic, proba_life_changing)) %>% 
  ggplot() +
  geom_pointinterval(aes(x = proba_life_changing, xmax = proba_life_changing.upper, xmin = proba_life_changing.lower, y = nootropic)) + 
  theme(legend.position="none") +
  xlab("Probability of being life-changing") + 
  ylab("")

ggsave("analysis/plots/antidepressant_life_changing.jpeg", width=10, height=6, units = "in", limitsize = F, dpi=300)



df_issues %>% 
  left_join(df_metadata %>% select(nootropic, type)) %>% 
  # mutate(type = if_else(nootropic == "P21" | nootropic == "BPC-157", "Peptides", type)) %>% 
  filter(type == "Anti-depressants" | nootropic %in% anti_depressant_list) %>% 
  filter(variant == "stan model") %>% 
  #filter(issue %in% c("side_effects", "long_term_side_effects")) %>% 
  #mutate(nootropic = fct_reorder(nootropic, prop)) %>% 
  ggplot() +
  geom_pointrange(aes(x = prop, xmax = prop_high, xmin=prop_low, y=nootropic, color=issue), position = position_dodge2(width=0.3))+
  xlab("Probability of issue") + 
  ylab("")

ggsave("analysis/plots/antidepressant_issues.jpeg", width=10, height=7, units = "in", limitsize = F, dpi=300)

View(df %>% 
  mutate(ssc = userID < 10000) %>% 
  filter(ssc == F) %>% 
  #filter(itemID %in% "Tianeptine") %>% 
  group_by(itemID) %>% 
  summarise(mean = mean(rating), median = median(rating), count = n()))



df
