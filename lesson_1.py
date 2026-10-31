prices =[100,102,99]
for t in range(1,len(prices)) :
    previes_price = prices[t-1]
    current_price = prices[t]
    daily_return = (current_price/previes_price)-1

    print(f"day {t}: {daily_return : .2%}")

 