import pandas as pd 
from random import shuffle
from math import log
from graphviz import Digraph
import time
styles = {
    'leaf': {'shape': 'rect', 'style': 'filled', 'color': 'yellow'},
    'crit': {'shape': 'rect'},
}


class split:
    #Initialisierung des Splits
    def __init__(self, attribute, values , split_type, bound = None):
        
        self.split_attribute = attribute      # Name des Attributes  
        self.split_values = values            # Menge der Split values 
        self.split_type = split_type          # Einer der Werte 'categorical' oder 'numerical'
        self.split_bound = bound              # Wird nur angegeben  wenn  split_type == numerical 
        pass
     
    def status(self):
        print('\n Attribut:',self.split_attribute,'\n split_values:', self.split_values)
        pass
    
    def copy(self):
        copy_split = split(self.split_attribute,self.split_values.copy(), self.split_type, self.split_bound )
        
        return copy_split
        pass
    
    pass


class node:
     #Initialisierung des Nodes
    def __init__(self, nNr = None, nLabel = None, nType = None, data = None, n_split = None):
        
        self.node_nr = nNr
        self.node_label = nLabel       
        self.node_type = nType           # eine der Ausprägungen 'criterion'/'leaf'     
        self.subset = data  
        self.node_split = n_split        # 'categorical' oder 'numerical' (nur angeben, wenn node_type = 'criterion')
        
        self.parent = None
        self.child_nodes = []
        self.edges = []
    
    def status(self):
        children = []
        for i in self.child_nodes:
            children.append(i.node_nr)
        
        print('\n Nr:',self.node_nr, '\n Label:', self.node_label,'\n Type:', self.node_type, '\n Children:', children, '\n Data: \n', self.subset )
        
    def copy(self):
        
        copy_node = node(self.node_nr, self.node_label, self.node_type, self.subset.copy(), self.node_split)
        copy_node.parent = self.parent
        
        for nd in self.child_nodes:
            copy_node.child_nodes.append(nd.copy())
            copy_node.child_nodes[-1].parent = copy_node
        
        for ed in self.edges:
            copy_node.edges.append(ed.copy()) 
        
        return copy_node
    pass


class edge:
    #Initialisierung 
    def __init__(self, root_nr = None, target_nr = None, label = ''):
        self.root_nr = root_nr
        self.target_nr = target_nr       
        self.label = label     
    
    def status(self):
        print('\n Root:',self.root_nr,'\n Target:', self.target_nr, '\n Label:', self.label)
        
    def copy(self):
        copy_edge = edge(self.root_nr, self.target_nr, self.label)
        
        return copy_edge
    
    pass


def find_all_splits(df_inputs, target_variable):
    
    # Wir wollen alle möglichen Splits in einer Liste sammeln
    # Für jedes Attribut wird ein Split erstellt und in list_of_splits abgespeichert
    list_of_splits = [] 
    
    attribute_list_categorical = df_inputs.drop(target_variable, axis = 1).select_dtypes(exclude = 'number').columns
    attribute_list_numerical = df_inputs.drop(target_variable, axis = 1).select_dtypes(include = 'number').columns
    
    
    # Wir gehen der Reihe nach alle kategorialen Attribute durch
    for current_attribute in attribute_list_categorical:            
        
        # Die verschiedenen Werte in der Spalte des aktuellen Attributs... 
        # ... werden mittels der unique()-Funktion ausgelesen
        value_set = df_inputs[current_attribute].dropna().unique()                                                      
        
        # Der aktuelle Split wird erstellt und er besteht aus Split-Attribut und Wertemenge 
        current_split = split(current_attribute, value_set, 'categorical') 
        
        list_of_splits.append(current_split)
        
        
     # Wir gehen der Reihe nach alle numerischen Attribute durch
    for current_attribute in attribute_list_numerical:            
    
        value_set = df_inputs[current_attribute].dropna().sort_values().unique()                                                      
        
        for i in range(len(value_set) - 1):
            current_bound = (value_set[i] + value_set[i+1]) / 2
            split_values = ['≤' + str(round(current_bound,2)), '>' + str(round(current_bound,2)) ]
            current_split = split(current_attribute, split_values, 'numerical', current_bound) 
            list_of_splits.append(current_split)        
                    
        #shuffle(list_of_splits)  

    return list_of_splits
    #Output: List of splits in the form --> 1 Split: [split_attribute, split_values]
    pass

    

def distribution(df_input, target_variable):
    
    return df_input[target_variable].value_counts().sort_index().tolist()


def misclassification_error(frequencies):
    
    return (sum(frequencies) - max(frequencies)) / sum(frequencies)

#Entropiefunktion, Eingabe: absolute Häufigkeiten der einzelnen Werte der Zielvariable als Liste
def entropy(frequencies):
    try:
        total = sum(frequencies) 
        return sum([-p / total * log(p / total, 2) for p in frequencies])
    except:
        return 0


def information_gain(df_inputs, target_variable, current_split):
    
    pre_entropy = entropy(distribution(df_inputs, target_variable))
    
    distributions_of_subsets = []
  
    if current_split.split_type == 'categorical':
        for split_value in current_split.split_values:

            current_df = df_inputs[df_inputs[current_split.split_attribute] == split_value]
            distributions_of_subsets.append(distribution(current_df, target_variable))
   
    elif current_split.split_type == 'numerical': 
        
            subset1 = df_inputs[df_inputs[current_split.split_attribute] <= current_split.split_bound]
            distributions_of_subsets.append(distribution(subset1, target_variable))
            
            subset2 = df_inputs[df_inputs[current_split.split_attribute] > current_split.split_bound]
            distributions_of_subsets.append(distribution(subset2, target_variable))
    
                 
    total_elements = df_inputs[target_variable].count()
    post_entropy = 0
    
    for current_distribution in distributions_of_subsets:
        elements = sum(current_distribution)
        post_entropy += (elements/total_elements)*entropy(current_distribution)
    
    info_gain = pre_entropy - post_entropy
    
    return info_gain



def information_gain_ME(df_inputs, target_variable, current_split):
    
    pre_ME = misclassification_error(distribution(df_inputs, target_variable))
    
    distributions_of_subsets = []
  
    if current_split.split_type == 'categorical':
        for split_value in current_split.split_values:

            current_df = df_inputs[df_inputs[current_split.split_attribute] == split_value]
            distributions_of_subsets.append(distribution(current_df, target_variable))
   
    elif current_split.split_type == 'numerical': 
        
            subset1 = df_inputs[df_inputs[current_split.split_attribute] <= current_split.split_bound]
            distributions_of_subsets.append(distribution(subset1, target_variable))
            
            subset2 = df_inputs[df_inputs[current_split.split_attribute] > current_split.split_bound]
            distributions_of_subsets.append(distribution(subset2, target_variable))
    
                 
    total_elements = df_inputs[target_variable].count()
    post_ME = 0
    
    for current_distribution in distributions_of_subsets:
        elements = sum(current_distribution)
        post_ME += (elements/total_elements)*misclassification_error(current_distribution)
    
    info_gain = pre_ME - post_ME
    
    return info_gain

def identify_best_split(df_inputs, target_variable, list_of_splits, criterion = information_gain):
    
    best_split = None
    best_information_gain = -1
    
    for current_split in list_of_splits:
            
            current_information_gain = criterion(df_inputs, target_variable, current_split)		
            
            if current_information_gain >= best_information_gain:
                best_information_gain = current_information_gain
                best_split = current_split
            
    return best_split
    #Output: Most productive split according to information gain
    pass

def identify_best_split2(df_inputs, target_variable, list_of_splits, criterion = information_gain):
    
    best_split = None
    best_information_gain = -1
    
    attribute_list_categorical = df_inputs.drop(target_variable, axis = 1).select_dtypes(exclude = 'number').columns
    attribute_list_numerical = df_inputs.drop(target_variable, axis = 1).select_dtypes(include = 'number').columns
    
    # Wir gehen der Reihe nach alle kategorialen Attribute durch
    for current_attribute in attribute_list_categorical:            
        
        # Die verschiedenen Werte in der Spalte des aktuellen Attributs... 
        # ... werden mittels der unique()-Funktion ausgelesen
        value_set = df_inputs[current_attribute].dropna().unique()                                                      
        
        # Der aktuelle Split wird erstellt und er besteht aus Split-Attribut und Wertemenge 
        current_split = split(current_attribute, value_set, 'categorical') 
        
        current_information_gain = criterion(df_inputs, target_variable, current_split)     
            
            if current_information_gain >= best_information_gain:
                best_information_gain = current_information_gain
                best_split = current_split
        
        
     # Wir gehen der Reihe nach alle numerischen Attribute durch
    for current_attribute in attribute_list_numerical:            
    
        value_set = df_inputs[current_attribute].dropna().sort_values().unique()                                                      
        
        for i in range(len(value_set) - 1):
            current_bound = (value_set[i] + value_set[i+1]) / 2
            split_values = ['≤' + str(round(current_bound,2)), '>' + str(round(current_bound,2)) ]
            current_split = split(current_attribute, split_values, 'numerical', current_bound) 
            
            current_information_gain = criterion(df_inputs, target_variable, current_split)     
            
            if current_information_gain >= best_information_gain:
                best_information_gain = current_information_gain
                best_split = current_split
            
            
            
    return best_split
    #Output: Most productive split according to information gain
    pass

def apply_split(df_inputs, current_split):
    
    list_of_subsets = []
    
    if current_split.split_type == 'categorical':
        for current_value in current_split.split_values:

            current_df = df_inputs[df_inputs[current_split.split_attribute] == current_value]     #NaN Werte fallen weg
            #current_df.append(df_inputs[df_inputs[current_split.split_attribute].isna()])        #Alternativ: Umgang mit NaN einfach in jeden Ast weiterleiten                  
            list_of_subsets.append(current_df)
            
    elif current_split.split_type == 'numerical':
        subset1 = df_inputs[df_inputs[current_split.split_attribute] <= current_split.split_bound] #NaN Werte fallen weg
        list_of_subsets.append(subset1)
            
        subset2 = df_inputs[df_inputs[current_split.split_attribute] > current_split.split_bound]
        list_of_subsets.append(subset2)
            
    
    return list_of_subsets
    #Output: List of Subsets(DataFrames)
    pass


def majority_value(df_inputs, target_variable):
    try:
        majority_value = df_inputs[target_variable].value_counts().idxmax() 
    except:
        majority_value = None
    return majority_value        
    pass

class DecisionTree:
    
    #Initialisierung des Decision Tree
    def __init__(self, crit = None, target = None, data = None ):
        
        self.tree_edges = []
        self.tree_nodes = {}
        self.tree_graph = Digraph()
        self.target = target
        self.data = data

        if (crit == None) or (crit == 'entropy'):
        	self.criterion = information_gain
        elif crit == 'misclassification_error':
        	self.criterion = information_gain_ME

        #self.depth = 0
        
        pass
    
    def grow_tree(self, df_input, target_variable, crit = 'entropy', max_depth = float('inf'), act_depth = 0):
        
        self.target = target_variable
        if crit == 'entropy':
        	self.criterion = information_gain
        elif crit == 'misclassification_error':
        	self.criterion = information_gain_ME

        #if self.depth < (act_depth + 1):
        #	self.depth = act_depth + 1
        attributes = (df_input.columns).drop(target_variable)
        #print('loading...')
        
        if act_depth >= max_depth:
            # Falls maximale Tiefe des Baums erreicht ist gib Leaf-Knoten mit Mehrheitswert aus Wert aus
            self.return_leaf_node(df_input, target_variable)

        # Überprüfen, ob mehr als ein Wert für die Zielvariable vorliegt
        elif df_input[target_variable].nunique() == 1:
            
            # Falls die Werte der Zielvariable im Datensatz alle gleich sind gib Leaf-Knoten mit dem Wert aus 
            self.return_leaf_node(df_input, target_variable)

            pass
        
        # Überprüfen ob noch Attribute für weitere Splits übrig sind
        elif len(attributes) == 0:
            
            #Falls Anzahl der Attribute 0 ist, gib ein leaf mit dem Mehrheitswert aus
            self.return_leaf_node(df_input, target_variable)
            
            pass
        
        # Falls vorherige Abfragen nicht zutrafen wird ein weiterer Split gesucht um ihn anzuwenden
        else:
            #Finde alle möglichen Splits
            list_of_splits = find_all_splits(df_input, target_variable)
            
            if len(list_of_splits) > 0:
                #Identifiziere den besten Split unter allen Splits
                best_split = identify_best_split(df_input, target_variable, list_of_splits, self.criterion)
                

                # Überprüfen: Ist der Split produktiv?
                if information_gain(df_input, target_variable, best_split) > 0: # Ist best_split produktiv?

                    #Wende den besten Split auf die Inputdaten an und erstelle somit ein Liste von Teildatensätzen
                    list_of_subsets = apply_split(df_input, best_split)

                    
                    #Den erstellten Split als Knoten ausgeben, falls best_split produktiv ist        
                    current_node = self.return_split_node(best_split, df_input)


                    #Rekursive weitere Anwendung für jeden erstellten Teildatensatz
                    for i in range(len(list_of_subsets)):
                        
                        next_node_nr = len(self.tree_nodes) + 1                        
                        new_input_subset = list_of_subsets[i].drop(best_split.split_attribute, axis = 1)

                        self.grow_tree(new_input_subset, target_variable, self.criterion, max_depth, act_depth+1)
                        
                        self.new_edge(root = current_node.node_nr, target = next_node_nr, label = best_split.split_values[i])
                    


                else:
                    #Falls best_split nicht produktiv, dann leaf ausgeben
                    self.return_leaf_node(df_input, target_variable)
                    pass
                
            else:
                #Falls list_of_splits leer ist gib einen 'leaf-node' aus #(Passiert, falls nur noch Split Attribute verbleiben 
                self.return_leaf_node(df_input, target_variable)
                pass    
        pass
    
    
    def return_leaf_node(self, data, target_variable):
        
        try: 
            node_nr = list(self.tree_nodes)[-1] + 1       # Neuer Knoten bekommt die nächst freie Nummer in tree_nodes
        except:
            node_nr = 1

        node_label = majority_value(data, target_variable) #Neuer Knoten bekommt als Label den Mehrheitswert für target_variable

        current_node = node(node_nr, node_label, 'leaf', data) # Knoten wird erstellt
        
        self.tree_nodes[node_nr] = current_node       # Knoten wird zum dictionary aller Knoten hinzugefügt
    
    
    
    def return_split_node(self, best_split, data):
        try: 
            node_nr = list(self.tree_nodes)[-1] + 1       # Neuer Knoten bekommt die nächst freie Nummer in tree_nodes
        except:
            node_nr = 1

        node_label = best_split.split_attribute       # Das Label des Knotens ist das aktuelle Split Attribut
        
        current_node = node(node_nr, node_label, 'criterion', data, best_split) # Knoten wird erstellt
        
        self.tree_nodes[node_nr] = current_node       # Knoten wird zum dictionary aller Knoten hinzugefügt
        
        return self.tree_nodes[node_nr]

    
    def new_edge(self, root, target, label):
        
        #Edge bauen
        new_edge = edge(root, target, str(label))                                            # N E U
        self.tree_edges.append(new_edge)
        
        
        #Nodes informieren
        nd_root = self.tree_nodes[root]
        nd_target = self.tree_nodes[target]
            
        nd_root.child_nodes.append(nd_target)
        nd_root.edges.append(new_edge)
        
        nd_target.parent = nd_root
        nd_target.edges.append(new_edge)
    
    
    
    def query(self, input_series):
        #Input: Series, die einen Wert für jedes Attribut enthält
        
        current_node = self.tree_nodes[1]
        next_nr = current_node.node_nr
        
        # Wir gehen so lange durch den Baum, bis wir in einem 'leaf-node' sind
        while current_node.node_type == 'criterion':
            
            #Prüfwert um später zu schauen, ob eine neuer Knoten gefunden wurde (wird nicht gefunden falls NaN)
            old_nr = next_nr
            
            #prüfen ob der Split 'categorical' oder 'numerical' ist
            if current_node.node_split.split_type == 'categorical':
    
                # Suche die Kante, die am aktuellen Knoten liegt und zum Wert der Input Series passt
                for edge in current_node.edges: 
                    if (edge.label == str(input_series[current_node.node_label])):
                        next_nr = edge.target_nr
                        break
                
                #Prüfen ob neuer Knoten gefunden wurde
                #Falls der im Kriterum abgefragte Wert fehlt (NaN) kann kein Ast ausgewählt werden
                #der Baum gibt den Mittelwert/Mehrheitswert im aktuellen Kriteriums-Knoten aus 
                if old_nr == next_nr:
                    return str(majority_value(current_node.subset, self.target))
                    
                
                # der nächste Knoten wir gesucht und als current_node gespeichert, um in die nächste Iteration der Schleife zu gehen 
                for nd in current_node.child_nodes:
                    if nd.node_nr == next_nr:
                        current_node = nd
                        break
            
            elif current_node.node_split.split_type == 'numerical':
                
                if (input_series[current_node.node_label] <= current_node.node_split.split_bound):
                    current_node = current_node.child_nodes[0]
                else:
                    current_node = current_node.child_nodes[1]

          
        return current_node.node_label
        
        pass

    def prediction_accuracy2(self, df_input, detailed = False, current_node=None):

        #NaN Zeilen werden extrahiert, da diese anders behandelt werden müssen
        df_nan = df_input[df_input.isnull().any(axis=1)]
        df_input = df_input[df_input.isnull().any(axis=1) == False]

        if current_node == None:
            current_node = self.tree_nodes[1]

        if (current_node.node_type == 'leaf') or (len(df_input) == 0):
            return df_input[self.target] == current_node.node_label
        
        predictions = pd.Series([])
        
        if current_node.node_split.split_type == 'categorical':
            for i in current_node.node_split.split_values:
                for j in current_node.child_nodes:
                    if j.node_split.split_attribute == i:
                        next_node = j
            predictions = predictions.append(self.prediction_accuracy2(df_input[df_input[current_node.node_split.split_attribute] == i], detailed=detailed, current_node = next_node))
        else:
            bound = current_node.node_split.split_bound
            child_node1 =  current_node.child_nodes[0]
            child_node2 =  current_node.child_nodes[1]
            predictions = predictions.append(self.prediction_accuracy2(df_input[df_input[current_node.node_split.split_attribute]<=bound], detailed=detailed, current_node = child_node1))
            predictions = predictions.append(self.prediction_accuracy2(df_input[df_input[current_node.node_split.split_attribute]>bound], detailed=detailed, current_node = child_node2))
        
        if current_node.node_nr == 1:
            
            #NaN-Zeilen mit der query-Funktion berücksichtigen
            targets = df_nan[self.target]
            prediction_list = []
            for i in range(len(df_nan)):
                prediction_list.append(self.query(df_input.iloc[i]))
            nan_predictions = pd.Series(prediction_list, name = 'prediction')
            nan_predictions.index = targets.index
            
            predictions = predictions.append(nan_predictions == targets) 

            #Ausgabe
            print(len(predictions))
            return predictions.mean()
        else:
            return predictions

        
    
    def prediction_accuracy(self, df_input, detailed = False):
        #print('loading...')
        targets = df_input[self.target]


        prediction_list = []
        
        #berechne alle outputs/predictions
        for i in range(len(df_input)):
            prediction_list.append(self.query(df_input.iloc[i]))

        predictions = pd.Series(prediction_list, name = 'prediction')
        predictions.index = targets.index
        

        accuracy = (predictions == targets).mean()
        print(len(predictions))
        if not detailed == True:
            return accuracy
        
        else:
            #Berechne Fehler verschiedener Art und gebe in crosstable aus
            df_evaluation = pd.concat([targets, predictions], axis = 1)

            values = targets.unique()
            columns= pd.Series(values, name = 'prediction')
            index = pd.Series(values, name = 'correct')
            df_crosstable = pd.DataFrame(columns=columns, index= index)

            for i in values:
                current_eval = df_evaluation[df_evaluation[self.target] == i]
                for j in values:
                    current_rate = (current_eval['prediction'] == j).mean()
                    df_crosstable[j][i] = current_rate

            return df_crosstable
        
        pass  
  
    
    
    def prune_node(self, prune_node_nr, prune_node = None):
        
        if prune_node == None:
            prune_node = self.tree_nodes[prune_node_nr]
            
                    
        if prune_node.parent == None:
            root_node = node()
        else:
            root_node = prune_node.parent
        
        list_of_children = prune_node.child_nodes
        prune_node.child_nodes = []
                
        if (prune_node.node_type == 'leaf') & (root_node.node_type == 'leaf'):
            
            del self.tree_nodes[prune_node_nr]
            
            for edge in prune_node.edges:
                if edge.target_nr == prune_node_nr:
                    self.tree_edges.remove(edge)
                    #root_node.edges.remove(edge)                                        # Warum geht das nicht so?
                    for ed in root_node.edges:
                        if ed.target_nr == prune_node_nr:
                            root_node.edges.remove(ed)
                    
        
        else:
            prune_node.node_type = 'leaf'
            prune_node.node_label = majority_value(prune_node.subset, self.target)
            
            
            for child in list_of_children:
                self.prune_node(child.node_nr, child)
                
            
            if root_node.node_type == 'leaf':
                del self.tree_nodes[prune_node_nr]
            
                for edge in prune_node.edges:
                    if edge.target_nr == prune_node_nr:
                        self.tree_edges.remove(edge)
                        #root_node.edges.remove(edge)
                        for ed in root_node.edges:
                            if ed.target_nr == prune_node_nr:
                                root_node.edges.remove(ed)
               
        pass
    
    def validation_pruning(self, validation_sample, root_node = None):                     #root_nr = 1
        
        if root_node == None:
            current_node = self.tree_nodes[1]
        else:
            current_node = root_node

        if current_node.node_type == 'leaf':
            
            pass

        else:    

            for child in current_node.child_nodes:                                                                    
                test_tree = self.validation_pruning(validation_sample, child)

            print('Test Node:', current_node.node_nr)
            
            if test_tree == None:
                test_tree = self.copy()
            
            pre_test_accuracy = test_tree.prediction_accuracy(validation_sample)
            
            test_tree.prune_node(current_node.node_nr)

            post_test_accuracy = test_tree.prediction_accuracy(validation_sample)
            
            if post_test_accuracy >= pre_test_accuracy:
                self.prune_node(current_node.node_nr, current_node)
                print('Prune Node:',current_node.node_nr)
                print('Node-Count', len(self.tree_nodes))
                print('New Test-Score', post_test_accuracy)
                return test_tree
                
            return None
        pass
    
    def manual_split(self, predictor_variable, node_nr = 1):
        
        if node_nr == 1:
            try:
                self.tree_nodes[1]
            except:
                self.tree_nodes[1] = node( nNr = 1, data = self.data)


        if self.tree_nodes[node_nr].node_type == 'criterion':
            self.prune_node(node_nr)

        self.tree_nodes[node_nr].node_type = 'criterion'
        self.tree_nodes[node_nr].node_label = predictor_variable
        if self.tree_nodes[node_nr].subset[predictor_variable].dtype in ['float64','int64']:
            
            list_of_splits = find_all_splits(self.tree_nodes[node_nr].subset[[self.target,predictor_variable]], self.target)
            best_split = identify_best_split(self.tree_nodes[node_nr].subset, self.target, list_of_splits, self.criterion)
            list_of_subsets = apply_split(self.tree_nodes[node_nr].subset, best_split)
            
            for i in range(len(list_of_subsets)):
                next_node_nr = list(self.tree_nodes)[-1] + 1
                self.return_leaf_node(list_of_subsets[i].drop(best_split.split_attribute, axis = 1), self.target)
                self.new_edge(root = node_nr, target = next_node_nr, label = best_split.split_values[i])

        else:
            value_set = self.tree_nodes[node_nr].subset[predictor_variable].dropna().unique()                                                       
            best_split = split(predictor_variable, value_set, 'categorical')
            list_of_subsets = apply_split(self.tree_nodes[node_nr].subset, best_split)
            
            for i in range(len(list_of_subsets)):
                next_node_nr = list(self.tree_nodes)[-1] + 1
                self.return_leaf_node(list_of_subsets[i].drop(best_split.split_attribute, axis = 1), self.target)
                self.new_edge(root = node_nr, target = next_node_nr, label = best_split.split_values[i])

        self.node_split = best_split        
        

        # NICHT FERTIG

        pass

    def reset_node_index(self):
        
        # NICHT FERTIG
        # edges müssen noch über Änderungen informiert werden (root, target) 

        tree_nodes_new = {}
        j = 0
        for i in self.tree_nodes:
            j +=1
            current_node = self.tree_nodes[i].copy()
            current_node.node_nr = j
            tree_nodes_new[j] = current_node

        self.tree_nodes = tree_nodes_new

        pass

    
    def print_tree(self):
        
        self.tree_graph = Digraph(filename = ('Tree_' + self.target))
        for current_node in list(self.tree_nodes.values()):
            if current_node.node_type == 'criterion':
                self.tree_graph.node(str(current_node.node_nr), str(current_node.node_label) + '?' + '\n Nr.:' + str(current_node.node_nr) + '\n' + str(distribution(current_node.subset, self.target)), styles['crit'])
            elif current_node.node_type == 'leaf':
                self.tree_graph.node(str(current_node.node_nr), str(current_node.node_label) + '\n Nr.:' + str(current_node.node_nr) + '\n' + str(distribution(current_node.subset, self.target)), styles['leaf'])    

        for current_edge in self.tree_edges:
                self.tree_graph.edge(str(current_edge.root_nr), str(current_edge.target_nr), current_edge.label)

        #self.tree_graph.view()
        
        
        pass
        
    def copy(self, copy_tree = None, current_node = None):
        if copy_tree == None:
            copy_tree = DecisionTree()
            copy_tree.target = self.target
            current_node = self.tree_nodes[1].copy()
        
        copy_tree.tree_nodes[current_node.node_nr] = current_node
        for edge in current_node.edges:
            if edge.target_nr == current_node.node_nr:
                copy_tree.tree_edges.append(edge)
                
        if len(current_node.child_nodes) == 0:
            pass
        else:
            for nd in current_node.child_nodes:
                copy_tree = self.copy(copy_tree, nd)
        
        
        return copy_tree
        pass
        
    