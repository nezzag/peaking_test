library(conflicted)
library(tidyverse)
library(osem)
conflicted::conflicts_prefer(dplyr::filter)

source("scripts/osem_prep_script.R")


co2_raw <- read_csv("data/processed/gcb_hist_co2.csv") %>%
  pivot_longer(cols = -c(region, region_name, variable,unit), names_to = "year", values_to = "co2", names_transform = as.numeric)

gdp_raw <- read_csv("data/processed/wdi_gdp.csv") %>%
  pivot_longer(cols = -c(region, region_name, variable,unit), names_to = "year", values_to = "gdp", names_transform = as.numeric)

energy_raw <- read_csv("data/processed/primary_energy.csv") %>%
  pivot_longer(cols = -c(region, region_name, variable,unit), names_to = "year", values_to = "prim_ener", names_transform = as.numeric)


remove_from_ROW <- c("Africa Eastern and Southern",
                     "Africa Western and Central", "Arab World",
                     "Central Europe and the Baltics", "Caribbean small states",
                     "East Asia & Pacific (excluding high income)",
                     "Early-demographic dividend", "East Asia & Pacific", "Europe & Central Asia (excluding high income)",
                     "Europe & Central Asia", "Euro area",
                     "European Union", "Fragile and conflict affected situations", "High income",
                     "Heavily indebted poor countries (HIPC)",
                     "IBRD only", "IDA & IBRD total",
                     "IDA total", "IDA blend", "IDA only", "Not classified",
                     "Latin America & Caribbean (excluding high income)", "Latin America & Caribbean",
                     "Least developed countries: UN classification", "Low income",
                     "Lower middle income", "Low & middle income",
                     "Late-demographic dividend","Middle East, North Africa, Afghanistan & Pakistan",
                     "Middle income", "Middle East, North Africa, Afghanistan & Pakistan (excluding high income)",
                     "North America", "OECD members", "Other small states", "Pre-demographic dividend",
                     "Pacific island small states",
                     "Post-demographic dividend", "Sub-Saharan Africa (excluding high income)",
                     "Sub-Saharan Africa", "Small states", "East Asia & Pacific (IDA & IBRD countries)",
                     "Europe & Central Asia (IDA & IBRD countries)",
                     "Latin America & the Caribbean (IDA & IBRD countries)",
                     "Middle East, North Africa, Afghanistan & Pakistan (IDA & IBRD)",
                     "South Asia (IDA & IBRD)", "Sub-Saharan Africa (IDA & IBRD countries)",
                     "Upper middle income", "World")

remove_from_ROW_code <- read_delim("data/raw/WorldBank_GDP.csv", skip = 3, delim = ",") %>%
  filter(`Country Name` %in% remove_from_ROW) %>%
  pull(`Country Code`)


# Remove aggregates -------------------------------------------------------

energy <- energy_raw %>%
  filter(!region %in% remove_from_ROW_code) %>%
  filter(!region_name %in% remove_from_ROW)

gdp <- gdp_raw %>%
  filter(!region %in% remove_from_ROW_code) %>%
  filter(!region_name %in% remove_from_ROW)

co2 <- co2_raw %>%
  filter(!region %in% remove_from_ROW_code) %>%
  filter(!region_name %in% remove_from_ROW)



# Plot data ---------------------------------------------------------------


energy %>%
  ggplot(aes(x = year, y = prim_ener, color = region_name)) +
  geom_line(linewidth = 1) +
  labs(x = "Year",
       y = "Region", color = "Region") +
  theme_minimal() +
  theme(legend.position = "none")


gdp %>%
  ggplot(aes(x = year, y = gdp, color = region_name)) +
  geom_line(linewidth = 1) +
  labs(x = "Year",
       y = "Region", color = "Region") +
  theme_minimal() +
  theme(legend.position = "none")


co2 %>%
  ggplot(aes(x = year, y = co2, color = region_name)) +
  geom_line(linewidth = 1) +
  labs(x = "Year",
       y = expression(CO[2]~"concentration (ppm)"), color = "Region",
       title = expression("Historical CO"[2]~"concentrations by region")) +
  theme_minimal() +
  theme(legend.position = "none")

# OSEM --------------------------------------------------------------------


osem_prep <- prepare_osem_co2_block(energy,
                                    gdp = gdp, co2 = co2,
                                    focus_countries = c("DEU", "CHN", "USA"), )

model <- run_model(
  specification = osem_prep$specification,
  input = osem_prep$local_data,
  dictionary = osem_prep$dictionary,
  max.ar = 2,
  trend = TRUE,
  use_logs = "both",
  saturation = c("TIS","IIS", "SIS"),
  saturation.tpval = 0.001
)

fcst_mod <- forecast_model(model,n.ahead = 5)

forecast_model(model,n.ahead = 5, exog_fill_method = "ets")




