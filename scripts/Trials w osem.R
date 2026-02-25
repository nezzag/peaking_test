library(conflicted)
library(tidyverse)
#devtools::install_github("moritzpschwarz/osem@extensions")
library(osem)
conflicted::conflicts_prefer(dplyr::filter)

source("scripts/osem_prep_script.R")


co2_raw <- read_csv("data/processed/gcb_hist_co2.csv") %>%
  pivot_longer(cols = -c(region, region_name, variable,unit), names_to = "year", values_to = "co2", names_transform = as.numeric)

gdp_raw <- read_csv("data/processed/wdi_gdp.csv") %>%
  pivot_longer(cols = -c(region, region_name, variable,unit), names_to = "year", values_to = "gdp", names_transform = as.numeric)

energy_raw <- read_csv("data/processed/primary_energy.csv") %>%
  pivot_longer(cols = -c(region, region_name, variable,unit), names_to = "year", values_to = "prim_ener", names_transform = as.numeric)

# Plot data ---------------------------------------------------------------

energy_raw %>%
  ggplot(aes(x = year, y = prim_ener, color = region_name)) +
  geom_line(linewidth = 1) +
  labs(x = "Year",
       y = "Region", color = "Region") +
  theme_minimal() +
  theme(legend.position = "none")


gdp_raw %>%
  ggplot(aes(x = year, y = gdp, color = region_name)) +
  geom_line(linewidth = 1) +
  labs(x = "Year",
       y = "Region", color = "Region") +
  theme_minimal() +
  theme(legend.position = "none")


co2_raw %>%
  ggplot(aes(x = year, y = co2, color = region_name)) +
  geom_line(linewidth = 1) +
  labs(x = "Year",
       y = expression(CO[2]~"concentration (ppm)"), color = "Region",
       title = expression("Historical CO"[2]~"concentrations by region")) +
  theme_minimal() +
  theme(legend.position = "none")

# OSEM --------------------------------------------------------------------

osem_prep <- prepare_osem_co2_block(energy = energy_raw,
                                    gdp = gdp_raw, co2 = co2_raw,
                                    focus_countries = c("DEU", "CHN", "USA"))



model <- run_model(
  specification = osem_prep$specification,
  input = osem_prep$local_data %>%
    mutate(values = case_when(na_item == "CO2_CHN" & time == as.Date("1899-01-01") ~ NA,
                              TRUE ~ values)),
  dictionary = osem_prep$dictionary,
  max.ar = 2,
  trend = TRUE,
  use_logs = "both",
  saturation = c("TIS","IIS", "SIS"),
  saturation.tpval = 0.001,
  constrain.to.minimum.sample = FALSE
)

fcst_mod <- forecast_model(model,n.ahead = 5)

forecast_model(model,n.ahead = 5, exog_fill_method = "ets")

fcst_insample <- forecast_insample(model, exog_fill_method = "ets", sample_share = 0.7)
plot(fcst_insample, first_date_insample_model = "2000-01-01")

save(fcst_insample, file = "data/processed/osem_fcst_insample.rda")
