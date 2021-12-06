import pandas as pd
import numpy as np

# df_ssc = pd.read_csv("../data/dataset_clean.csv")
# df_reddit = pd.read_csv("../data/nootropics_survey_reddit_converted.csv")
# df_darkha = pd.read_csv("../data/nootropics_survey_darkha_converted.csv")
# #
# print(df_ssc.groupby("itemID").aggregate(lambda x:sum(~pd.isnull(x))).sort_values(by="rating"))
# print("####")
# print(pd.isnull(df_reddit).sum(axis=0))
# print("######")
# print(pd.isnull(df_darkha).sum(axis=0))
#
#columns = set(df_ssc["itemID"]).union(set(df_darkha.columns)).union(set(df_reddit.columns))
# print("columns")
#print(np.sort(list(columns)))
#
# intersection_features = set(df_reddit.columns).intersection(set(df_darkha.columns))
# print(intersection_features)
# print(np.sort([col for col in df_reddit.columns if col not in intersection_features]))
#
# print(np.sort([col for col in df_darkha.columns if col not in intersection_features]))

# df_reddit.columns = ['ALCAR', 'Cerebrolysin', 'Dextroamphetamine (Speed)', 'Dihexa', 'Etifoxine',
#                      'Gingko, Biloba', 'IDRA-21', 'Inositol', 'Kratom', 'MAOI',
#                      'N-acetyl Cysteine (NAC)', 'N-methyl-cyclazodone', 'P21', 'Palmitoylethano', 'Phosphatidyl Serine', 'Pregenolone', 'Psilocybin Microdose', 'Methylphenidate (Ritalin)',
#                      'Tryptophan', 'Valerian Root', 'rgpu-95']
#
# df_darkha.columns = ['Adderall', 'Adrafinil', 'Agmatine', 'Aniracetam', 'Armodafinil', 'BPC-157',
#                      'Boswellia', 'Carnitine, /, Acetyl-L-Carnitine', 'Choline', 'Creatine'
# , 'Curcumin', 'DMHA', 'Doepezil', 'Fasoracetam', 'GABA', 'Ginkgo, Biloba'
# , 'Guanfacine', 'Kava', 'L-Deprenyl', 'MethyleneBlue', 'Methylphenidate, (Ritalin)'
# , 'N-Acetyl-L-Tyrosine', 'Nefiracetam', 'Pramiracetam', 'Sarcosine'
# , 'SelankandNASelanketc', "St, John's, Wort", 'Sulbutiamine', 'Sunifiram'
# , 'Tianeptine']


old_names = ['5-HTP', 'ALCAR', 'Adderall', 'Adrafinil', 'Agmatine', 'Alpha-GPC',
             "AlphaBrainproprietaryblend",
             'Aniracetam', 'Armodafinil', 'Ashwagandha', 'BPC-157', 'Bacopa',
             'Black Seed Oil', 'Boswellia', 'CBD', 'Caffeine',
             'Carnitine / Acetyl-L-Carnitine', 'Cerebrolysin', 'Choline', 'Coluracetam',
             'Creatine', 'Curcumin', "DMAE", 'DMHA', 'Dextroamphetamine (Speed)', 'Dihexa',
             'Doepezil', "Emoxypine", "Epicorasimmunebooster", 'Etifoxine', 'Fasoracetam', 'GABA', 'Gingko Biloba',
             'Ginkgo biloba', 'Ginseng', 'Guanfacine', 'Huperzine A', 'IDRA-21', 'Inositol',
             'Kava', 'Kratom', 'L-Deprenyl', 'LSD', "Lion's Mane", 'MAOI', 'Magnesium',
             'Melatonin', 'MethyleneBlue', 'Methylphenidate', 'Modafinil',
             'N-Acetyl-L-Tyrosine', 'N-acetyl Cysteine (NAC)', 'N-methyl-cyclazodone',
             'NSI-189', 'Nefiracetam', 'Nicotine', 'Noopept', 'Omega-3 Supplements',
             'Oxiracetam', 'P21', 'PRL853', 'Palmitoylethano', 'Phenibut',
             'Phenylpiracetam', 'Phosphatidyl Serine', "Picamilon", 'Piracetam', 'Pramiracetam',
             'Pregenolone', 'Psilocybin Microdose', 'Rhodiola', 'Ritalin LA', 'Sarcosine',
             'SelankandNASelanketc', 'Seligiline', 'SemaxandNASemaxetc', "St John's Wort",
             'Sulbutiamine', 'Sunifiram', 'Theanine', 'Tianeptine', 'Tryptophan',
             'Tyrosine', "Unifiram", 'Uridine', 'Valerian Root', 'rgpu-95']

new_names = ['5-HTP', 'ALCAR', 'Adderall', 'Adrafinil', 'Agmatine', 'Alpha-GPC', "AlphaBrainproprietaryblend",
             'Aniracetam', 'Armodafinil', 'Ashwagandha', 'BPC-157', 'Bacopa',
             'Black Seed Oil', 'Boswellia', 'CBD', 'Caffeine',
             'Carnitine / Acetyl-L-Carnitine', 'Cerebrolysin', 'Choline', 'Coluracetam',
             'Creatine', 'Curcumin', "DMAE", 'DMHA', 'Dextroamphetamine', 'Dihexa',
             'Doepezil', "Emoxypine", "Epicorasimmunebooster", 'Etifoxine', 'Fasoracetam', 'GABA', 'Ginkgo Biloba',
             'Ginkgo Biloba', 'Ginseng', 'Guanfacine', 'Huperzine-A', 'IDRA-21', 'Inositol',
             'Kava', 'Kratom', 'L-Deprenyl', 'LSD (microdose)', "Lion's Mane", 'MAOIs (except Selegiline)', 'Magnesium',
             'Melatonin', 'Methylene blue', 'Methylphenidate (Ritalin)', 'Modafinil',
             'N-Acetyl-L-Tyrosine', 'N-acetyl Cysteine (NAC)', 'N-methyl-cyclazodone',
             'NSI-189', 'Nefiracetam', 'Nicotine', 'Noopept', 'Omega-3 Supplements',
             'Oxiracetam', 'P21', 'PRL-8-53', 'Palmitoylethanolamide', 'Phenibut',
             'Phenylpiracetam', 'Phosphatidylserine', "Picamilon", 'Piracetam', 'Pramiracetam',
             'Pregenolone', 'Psilocybin (microdose)', 'Rhodiola', 'Methylphenidate (Ritalin)', 'Sarcosine',
             'Selank (or NA-Selank etc)', 'Selegiline', 'Semax (or NA-Semax etc)', "St John's Wort",
             'Sulbutiamine', 'Sunifiram', 'Theanine', 'Tianeptine', 'Tryptophan',
             'Tyrosine', "Unifiram", 'Uridine', 'Valerian Root', 'RGPU-95 (Cebaracetam...)']



short_names = ['5-HTP', 'ALCAR', 'Adderall', 'Adrafinil', 'Agmatine', 'Alpha-GPC', "AlphaBrainproprietaryblend",
             'Aniracetam', 'Armodafinil', 'Ashwagandha', 'BPC-157', 'Bacopa',
             'Black Seed Oil', 'Boswellia', 'CBD', 'Caffeine',
             'Carnitine', 'Cerebrolysin', 'Choline', 'Coluracetam',
             'Creatine', 'Curcumin', "DMAE", 'DMHA', 'Dextroamphetamine', 'Dihexa',
             'Doepezil', "Emoxypine", "Epicorasimmunebooster", 'Etifoxine', 'Fasoracetam', 'GABA', 'Ginkgo Biloba',
             'Ginkgo Biloba', 'Ginseng', 'Guanfacine', 'Huperzine-A', 'IDRA-21', 'Inositol',
             'Kava', 'Kratom', 'L-Deprenyl', 'LSD (micro)', "Lion's Mane", 'MAOIs', 'Magnesium',
             'Melatonin', 'Methylene blue', 'Methylphenidate', 'Modafinil',
             'N-Acetyl-L-Tyrosine', 'NAC', 'N-methyl-cyclazodone',
             'NSI-189', 'Nefiracetam', 'Nicotine', 'Noopept', 'Omega-3',
             'Oxiracetam', 'P21', 'PRL-8-53', 'Palmitoylethanolamide', 'Phenibut',
             'Phenylpiracetam', 'Phosphatidylserine', "Picamilon", 'Piracetam', 'Pramiracetam',
             'Pregenolone', 'Psilocybin (micro)', 'Rhodiola', 'Methylphenidate', 'Sarcosine',
             'Selank', 'Selegiline', 'Semax', "St John's Wort",
             'Sulbutiamine', 'Sunifiram', 'Theanine', 'Tianeptine', 'Tryptophan',
             'Tyrosine', "Unifiram", 'Uridine', 'Valerian Root', 'RGPU-95']

#indices = list(map(len, short_names))
#print(np.array(short_names)[np.argsort(indices)])

other_nootropics = ["Taurine", "Kanna (except Zembrin)", "Zembrin", "Shilajit", "Cordyceps", "Lemon balm",
                    "Nicotinamide riboside", "Nicotinamide mononucleotide", "Polygala tenuifolia",
                    "Maca", "Bromantane", "Niacin", "Saffron", "Glycine", "Berberine",
                    "White jelly mushrooms", "Theacrine (aka Teacrine)", "Methylliberine (aka Dynamine)",
                    "Red reishi mushrooms", "7,8-dihydroxyflavone", "9-MBC", "SSRIs (Prozac, Lexapro...)", "Bupropion (Wellbutrin, Zyban...)",
                    "Alprazolam (Xanax)", "SNRIs (Cymbalta, Effexor...)", "SAM-e", "Fermented drinks (Kefir, Kombucha...)", "Probiotics"]

short_other_nootropics = ["Taurine", "Kanna", "Zembrin", "Shilajit", "Cordyceps", "Lemon balm",
                    "Nicotinamide riboside", "Nicotinamide mononucleotide", "Polygala tenuifolia",
                    "Maca", "Bromantane", "Niacin", "Saffron", "Glycine", "Berberine",
                    "White jelly mushrooms", "Theacrine", "Methylliberine",
                    "Red reishi mushrooms", "7,8-dihydroxyflavone", "9-MBC", "SSRIs", "Bupropion", "Alprazolam (Xanax)", "SNRIs", "SAM-e", "Fermented drinks", "Probiotics"]

lifestyle_nootropics = ["Ketogenic diet", "Carnivore diet", "Vegetarian diet", "Vegan diet", "No Fap (or otherwise avoiding masturbation)",
                        "Bright lights in morning / Dawn simulator", "Trying to get more sleep",
                        "Trying to get less sleep", "Weightlifting", "Low intensity cardio (hiking, light jogging...)", "High intensity cardio (HIIT, soccer...)", "Meditation"]

short_lifestyle_nootropics = ["Ketogenic diet", "Carnivore diet", "Vegetarian diet", "Vegan diet", "No Fap", "Morning lights", "More sleep", "Less sleep",
                              "Weightlifting", "Low intensity cardio", "High intensity cardio", "Meditation"]

classic_nootropics = ["Rhodiola", "Aniracetam", "Phenibut", "Ashwagandha", "Bacopa", "Piracetam", "Choline",
                      "Noopept", "Adderall", "Nicotine", "Creatine", "Theanine", "Modafinil", "Melatonin", "Caffeine",
                      "Magnesium", "Ginseng", "CBD", "Omega-3 Supplements"]

assert len(list(set(new_names).union(set(other_nootropics)).union(lifestyle_nootropics).intersection(set(classic_nootropics)))) == len(classic_nootropics)

short_dic = {}
for i, nootropic in enumerate(new_names):
        short_dic[nootropic] = short_names[i]
for i, nootropic in enumerate(lifestyle_nootropics):
    short_dic[nootropic] = short_lifestyle_nootropics[i]
for i, nootropic in enumerate(other_nootropics):
    short_dic[nootropic] = short_other_nootropics[i]




weird_nootropics = list(set(new_names).union(set(other_nootropics)).difference(classic_nootropics))
classic_nootropics, lifestyle_nootropics, weird_nootropics = np.sort(list(set(classic_nootropics))), np.sort(list(set(lifestyle_nootropics))), np.sort(list(set(weird_nootropics)))
to_drop = ["AlphaBrainproprietaryblend", "Epicorasimmunebooster"]
weird_nootropics = [noot for noot in weird_nootropics if noot not in to_drop]
all_nootropics = np.sort(np.concatenate([classic_nootropics, lifestyle_nootropics, weird_nootropics]))


#def noot_type(noot):
 #   if noot in classic_nootropics:
 #       return "classic"
 #   if noot in lifestyle_nootropics:
 #       return "lifestyle"
 #   if noot in weird_nootropics:
#        return "other"
#new_df = pd.DataFrame({"nootropic":all_nootropics,
 #                      "nootropic_short":list(map(lambda x:short_dic[x], all_nootropics)),
 #                      "type":list(map(noot_type, all_nootropics))})

#df_links = pd.read_csv("../nootropics_metadata.csv", sep=";")
#print(df_links)

#new_df = new_df.merge(df_links, on = ["nootropic", "nootropic_short"], how="left")

#new_df.to_csv("../test.csv")
# import requests
# import urllib
# df = pd.read_csv("../test.csv", sep=";")
#
# wiki_links = []
# for i, row in df.iterrows():
#     if type(row["wiki_link"]) == float: #check for Nans
#         try:
#             urllib.request.urlopen("https://en.wikipedia.org/wiki/{}".format(row["nootropic"])).getcode()
#             wiki_links.append("https://en.wikipedia.org/wiki/" + row["nootropic"])
#             print("success wiki")
#
#         except:
#             try:
#                 urllib.request.urlopen("https://en.wikipedia.org/wiki/{}".format(row["nootropic_short"])).getcode()
#                 wiki_links.append("https://en.wikipedia.org/wiki/" + row["nootropic_short"])
#                 print("success wiki 2")
#             except:
#                 print("fail")
#                 wiki_links.append(np.nan)
#     else:
#         wiki_links.append(row["wiki_link"])
#
# examine_link = []
# examine_default = "https://examine.com/supplements/{}/"
#
# def check_examine_link(name):
#     redirect = ''
#     responses = requests.get(examine_default.format(name))
#     for response in responses.history:
#         redirect += response.url
#     print(examine_default.format(name))
#     print(redirect)
#     return len(redirect) == 0
#
# for i, row in df.iterrows():
#     if type(row["examine_link"]) == float: #check for Nans
#         if check_examine_link(row["nootropic"].lower()):
#             examine_link.append(examine_default.format(row["nootropic"].lower()))
#             print("yes")
#         else:
#             if check_examine_link(row["nootropic_short"].lower()):
#                 examine_link.append(examine_default.format(row["nootropic_short"].lower()))
#                 print("yes2")
#             else:
#                 print("no")
#                 examine_link.append(np.nan)
#     else:
#         examine_link.append(row["examine_link"])
#
# df["wiki_link"] = wiki_links
# df["examine_link"] = examine_link
# df.to_csv("../test2.csv")

# print(len(classic_nootropics))
# print(len(list(set(classic_nootropics).intersection(set(new_names)))))
# print(set(other_nootropics).intersection(set(new_names)))
# print(set(lifestyle_nootropics).intersection(set(new_names)))
# rosetta_dic = {}
# short_dic = {}
# print(len(old_names))
# print(len(new_names))
#
# for i, nootropic in enumerate(old_names):
#     if nootropic not in to_drop:
#         print(i)
#         print(nootropic)
#         print(new_names[i])
#         print("######")
#         rosetta_dic[nootropic] = new_names[i]
#         short_dic[nootropic] = short_names[i]
#
# print(rosetta_dic)
#
# df_ssc = pd.read_csv("../data/dataset_clean.csv")
#
# print(np.sort(np.unique(df_ssc["itemID"])))
# print(len(np.sort(np.unique(df_ssc["itemID"]))))
#
# df_ssc = df_ssc[~np.isin(df_ssc["itemID"], to_drop)]
#
# print(np.sort(np.unique(df_ssc["itemID"])))
# print(len(np.sort(np.unique(df_ssc["itemID"]))))
#
# df_ssc["itemID"] = list(map(lambda x:rosetta_dic[x], df_ssc["itemID"]))
#
# df_ssc.to_csv("../data/dataset_clean_right_names.csv", index=False)
#
# avalaible_nootropics = np.unique(df_ssc["itemID"])
#
# print(avalaible_nootropics)
#