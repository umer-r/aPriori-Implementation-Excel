from numpy import True_
from datetime import datetime
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Basic settings
currentDT = datetime.now()
filename = 'Orders.xlsx'
col_desc = 'Title'
col_cust = 'Customer'
col_val = 'Quantity'
outfile = 'combinations_analysis_'+currentDT.strftime("%d-%m-%Y_%H.%M.%S")+".xlsx"
outfile_1 = str(outfile)

# Algorithmic Variables
min_support_val = 0.005
lift_val = 6
confidence_val = 0.8

def encode_units(x):
    if x <= 0:
        return False
    if x >= 0:
        return True

def predict(antecedent, rules, max_results= 6):
    preds = rules[rules['antecedents'] == antecedent]
    preds = preds['consequents'].apply(iter).apply(next)
    preds = preds[:max_results].reset_index()
    preds = preds.to_string(header=None, index=False)
    return preds


if __name__ == "__main__":

    df = pd.read_excel(filename)
    df[col_desc] = df[col_desc].str.strip()
    df.dropna(axis=0, subset=[col_cust], inplace=True)
    df[col_cust] = df[col_cust].astype('str')
    df = df[~df[col_cust].str.contains('C')]


    basket = (df.groupby([col_cust, col_desc])[col_val].sum().unstack().reset_index().fillna(0).set_index(col_cust))

    basket_sets = basket.applymap(encode_units)

    #Filter the dataframe using standard pandas code, for a large lift (6) and high confidence (.8)
    frequent_itemsets = apriori(basket_sets, min_support=min_support_val, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules[ (rules['lift'] >= lift_val) & (rules['confidence'] >= confidence_val) ]



    print("\n\t\t\t\t+---------------------------------------+")
    print("\t\t\t\t|  Product Combination Analysis System  |")
    print("\t\t\t\t+---------------------------------------+")
    print("\t\t\t\t|       Umer M. | Github.com/umer-r     |")
    print("\t\t\t\t+---------------------------------------+\n\n")
    print(rules)

    user_var2 = input("\nGet recommendations for specific product? (Y/N) : ")

    if user_var2.lower() == "y":
        prod_name = input("Enter Name of product (same as in sheet) : ")
        preds = predict({prod_name}, rules)
        print("\n--Predictions Below--\n",preds)
    else:
        print('\nExit!')

    user_var1 = input("\nStore Results? (Y/N) : ")
    
    if user_var1.lower() == "y":
        rules['antecedents'] = rules['antecedents'].astype('string')
        rules['consequents'] = rules['consequents'].astype('string')
        rules['antecedents'] = rules['antecedents'].str.lstrip("frozenset({")
        rules['antecedents'] = rules['antecedents'].str.rstrip("})")
        rules['consequents'] = rules['consequents'].str.lstrip("frozenset({")
        rules['consequents'] = rules['consequents'].str.rstrip("})")
        rules.to_excel(outfile_1)
        print("File Saved: "+outfile_1)
    else:
        print('\nFile Not saved!')

    user_var3 = input("\nPress any key to exit....")