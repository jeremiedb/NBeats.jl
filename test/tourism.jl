using CSV
using DataFrames

df = DataFrame(CSV.File("data/tourism/monthly_in.csv"))
