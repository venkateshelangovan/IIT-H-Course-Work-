// importing the libraries
#include <iostream>
using namespace std;

// create a node 
class Tree_Node{
    public:
    int element; // element to be stored in the node 
    Tree_Node *left_ptr; // left pointer to the node 
    Tree_Node *right_ptr; // right pointer to the node 
    int height; // height of the tree from that node i.e max(left sub tree,right sub tree)+1 from that node
};

/*
2. Insert function 

Input : root (root node) , element(element to be inserted)
Returns : root (root node)

Description : This function takes the root of the tree and element to be inserted as input.The below function will insert 
the element into the tree and check whether it is balanced. If it is not balanced it rotates(Left rotation or
Right rotation) the tree depending on the tree structure at that point of time 
*/
Tree_Node *Insert(Tree_Node *root,int element);

/*
3. Delete function 
Input : root (root node) , element(element to be deleted)
Returns : root (root node)

Description :This function takes the root of the tree and element to be deleted as input. The below function will delete the 
element from the tree and replace it with inorder successor and check whether the tree is balanced and if it is
not balanced it will make it balance and return the root node
*/
Tree_Node *Delete(Tree_Node *root,int element);

/*
4. Query function

Input : root(root node),a,b 
Output : query_result

Description : This function takes the root node and starting and ending range of numbers (a,b) and returns the 
count of numbers in that closed interval(a,b) such that there are distinct elements x, y in the array (array is
created when the query is given as input) with x + y = t.
*/
int query(Tree_Node *root,int a,int b);

/*
5. create_node

Input : element (element to be added to the new node)
Returns : new_node(newly created node having the element that is being passed)

Description : This function takes the input as element and it creates the new node and assign the value as element which is 
passed to the function. Here tree node is created where element is stored and left and right pointers are made
NULL and height is initialized to 1 for this new node that is created.
*/
Tree_Node *create_node(int element);

/*
6. heightOfTheTree

Input : root (root node)
Returns : 0 if the root is NULL and root->height if the root is not NULL 

This function takes the root node as input and returns the height if the root is not null
*/
int heightOfTheTree(Tree_Node *root);

/*
7. max

Input : a ,b 
Returns : maximum among a and b 

Description : This functions takes the input of integers a and b and returns maximum among those two integers 
*/
int max(int a,int b);

/* 
8. AVLBalanceChecker

Input : root(root node)
Returns : 0 if root is NULL else it returns height of left sub tree - height of right sub tree

Description : This function takes the root as input and check whether the tree is balanced and it is calculated by 
tree_balance=height of left subtree - height of right sub tree
*/
int AVLBalanceChecker(Tree_Node *root);

/*
9. Balancing_Trees

Input : root(root node),tree_balance(output that we got from AVLBalanceChecker),element(element to be either inserted or delted)
Returns : root (root node) after the tree gets balanced

Description : This function takes the input such as root,tree balance factor and element(which could be either used for 
insertion or deletion). It performs rotation if the tree is unbalanced else it returns the root.This performs 
four types of rotations.Right rotation,right left rotation,left rotation and left right rotation depending on
the situation

*/
Tree_Node *Balancing_Trees(Tree_Node *root,int tree_balance,int element);

/*
10. RotateRight 
Input : root (root node)
Returns : updated root after right rotation is done 

Description : This function takes the root node as input and Right rotation is done .
*/
Tree_Node *RotateRight(Tree_Node *root);

/*
11. RotateLeft

Input : root (root node)
Returns : updated root after left rotation is done 

Description : This function takes the root node as input and Left rotation is done .
*/
Tree_Node *RotateLeft(Tree_Node *root);

/*
12 MinNode

Input : root (root node)
Returns : left most child from the root node 

This function takes the root node as input and returns the left most child from the root node being passed
*/
Tree_Node *MinNode(Tree_Node *root);

/* 
number_of_elements is used for counting number of elements in the array and it is used in CountElementsInAVLTree function.
It is initialized to 1 having root node by default. If root node is NULL the function CountElementsInAVLTree
will return 0 else it will use this count into consideration for root node on counting the number of elements
in the tree at that point of time.
*/
int number_of_elements=1; 
/*
13. CountElementsInAVLTree

Input : root(root node)
Returns : number_of_elements(number of values in the AVL Tree at that point of time)

Description : This function takes the root node as input and it returns the number of elements in the array at that point of
time when the function is called.
*/
int CountElementsInAVLTree(Tree_Node *root);

/*
ind is made 0 to store the sorted arr elements while doing the inorder traversal.It is resetted to 0 after
inorder traversal is done
*/
int ind=0;
/*
14. InOrderTraversal

Input : root(root node)
Performs : Performs inorder traversal and stores the element in the sorted order to the array

Description: This function takes the root node and array as input and returns the inorder traversal(returns the sorted array)
at this point of time.
*/
void InOrderTraversal(Tree_Node *root,int arr[]);

/*
15. BinarySearch

Input : arr[](sorted array),find(element to be found),s(start index of array),e(end index of array)
Returns : 1 if element found else -1 

Description : This function takes sorted array,element to be searched(find),first index of array and last index of array as
input and returns 1 if the element is found else it returns -1
*/
int BinarySearch(int arr[],int find,int s,int e);


// creating the insert,delete and query functions for AVL Tree 

// function to insert a number k into AVL Tree
Tree_Node *Insert(Tree_Node *root,int element){
    // If the root is null then insert the element
    if(root==NULL){
        return create_node(element); // this function creates the tree node with value as the element 
    }
    if(element<root->element){
        // checking left sub tree to insert the element
        root->left_ptr=Insert(root->left_ptr,element);
    }
    else{
        // checking right sub tree to insert the element
        root->right_ptr=Insert(root->right_ptr,element);
    }
    // updating the height of the tree in root 
    root->height=1+max(heightOfTheTree(root->left_ptr),heightOfTheTree(root->right_ptr));
    // check for balancing of tree in left and right side of the tree
    int tree_balance=AVLBalanceChecker(root);
    // if the tree is imbalanced we rotate the tree to make it balanced 
    root=Balancing_Trees(root,tree_balance,element);
    return root;
}

//function to delete the instance of given element from the AVL tree 
Tree_Node *Delete(Tree_Node *root,int element){
    if(root==NULL){
        return root;
    }
    // search for element to be deleted in left sub tree
    else if(element<root->element){
        root->left_ptr=Delete(root->left_ptr,element);
    }
    // search for element to be deleted in right sub tree 
    else if(element>root->element){
        root->right_ptr=Delete(root->right_ptr,element);
    }
    else {
        if ((root->left_ptr == NULL) ||(root->right_ptr == NULL)) {
            Tree_Node *temp = root->left_ptr ? root->left_ptr : root->right_ptr;
            if (temp == NULL) {
                temp = root;
                root = NULL;
                } 
                else{
                    *root = *temp;
                    free(temp);
                } 
        }
        else {
            // getting the inorder successor 
            Tree_Node *temp = MinNode(root->right_ptr);
            // storing the inorder successor in root node 
            root->element = temp->element;
            // delete the inorder successor 
            root->right_ptr = Delete(root->right_ptr,temp->element);
        }
    }
    if(root==NULL){
        return root;
    }
    // updating the height of the tree in root 
    root->height=1+max(heightOfTheTree(root->left_ptr),heightOfTheTree(root->right_ptr));
    // check for balancing of tree in left and right side of the tree
    int tree_balance=AVLBalanceChecker(root);
    // balance the tree and return the root if it is unbalanced
    root=Balancing_Trees(root,tree_balance,element);
    return root;   
}

//query function to print the number of target values in the closed interval [a,b]
int query(Tree_Node *root,int a,int b){
    int query_result=0;
    int n,search;
    //count the number of elements at this given point of query time
    n=CountElementsInAVLTree(root);
    number_of_elements=1;
    //creating the dynamic memory allocation of size n at this point of time when query is executed
    int* sorted_arr = new int[n];
    //doing inorder traversal and sorted array is stored in sorted_arr
    InOrderTraversal(root,sorted_arr);
    ind=0;
    int found;
    for(int x=a;x<=b;x++){
        found=0;
        for(int i=0;i<n-1;i++){
            search=x-sorted_arr[i];
            found=BinarySearch(sorted_arr,i+1,n-1,search);
            if(found==1){
                query_result+=found;
                break;
            }
        }
    }
    // after the query result is computed the array that we created is deleted 
    delete[] sorted_arr;
    return query_result;
}

// sub-functions to compute insert,delete and query functions


// creating the new Node 
Tree_Node *create_node(int element){
    Tree_Node *new_node=new Tree_Node();
    new_node->element=element;
    new_node->left_ptr=NULL;
    new_node->right_ptr=NULL;
    new_node->height=1;
    return new_node;
}

// function to return max of two integers 
int max(int a,int b){
    if(a<b){
        return b;
    }
    else{
        return a;
    }
}

// this function returns the balanced tree 
Tree_Node *Balancing_Trees(Tree_Node *root,int tree_balance,int element){
    if(tree_balance>1){
        if(element<root->left_ptr->element){
            return RotateRight(root);
        }
        else if(element>root->left_ptr->element){
            root->left_ptr=RotateLeft(root->left_ptr);
            return RotateRight(root);
        }
    }
    else if(tree_balance<-1){
        if(element<root->right_ptr->element){
            root->right_ptr=RotateRight(root->right_ptr);
            return RotateLeft(root);
        }
        else if(element>root->right_ptr->element){
            return RotateLeft(root);
        }
    }
    return root;
}

// height of the tree from the given root node
int heightOfTheTree(Tree_Node *root){
    return (root==NULL) ? 0: root->height;
}

//check for balanced binary search tree to verify whether tree is balanced or not 
int AVLBalanceChecker(Tree_Node *root){
    return (root==NULL) ? 0 : heightOfTheTree(root->left_ptr)-heightOfTheTree(root->right_ptr);
}

// to get the left mode child from the root node passed
Tree_Node *MinNode(Tree_Node *root){
    return ((root==NULL)||(root->left_ptr==NULL)) ? root : MinNode(root->left_ptr);
}

//Rotating the tree left 
Tree_Node *RotateLeft(Tree_Node *node){
    Tree_Node *l1=node->right_ptr;
    Tree_Node *l2=l1->left_ptr;
    l1->left_ptr=node;
    node->right_ptr=l2;
    node->height=1+max(heightOfTheTree(node->left_ptr),heightOfTheTree(node->right_ptr));
    l1->height=1+max(heightOfTheTree(l1->left_ptr),heightOfTheTree(l1->right_ptr));
    return l1;
}

// Rotating the tree right 
Tree_Node *RotateRight(Tree_Node *node){
    Tree_Node *r1=node->left_ptr;
    Tree_Node *r2=r1->right_ptr;
    r1->right_ptr=node;
    node->left_ptr=r2;
    node->height=1+max(heightOfTheTree(node->left_ptr),heightOfTheTree(node->right_ptr));
    r1->height=1+max(heightOfTheTree(r1->left_ptr),heightOfTheTree(r1->right_ptr));
    return r1;
}

// count the number of elements in AVL Tree when query is passed
int CountElementsInAVLTree(Tree_Node *root){
    if(root==NULL){
        return 0;
    }
    if(root->left_ptr!=NULL){
        number_of_elements+=1;
        number_of_elements=CountElementsInAVLTree(root->left_ptr);
    }
    if(root->right_ptr!=NULL){
        number_of_elements+=1;
        number_of_elements=CountElementsInAVLTree(root->right_ptr);
    }
    return number_of_elements;
}

// inorder traversal to get the sorted elements till this query point
void InOrderTraversal(Tree_Node *root,int arr[]){
    if(root==NULL){
        return;
    }
    InOrderTraversal(root->left_ptr,arr);
    arr[ind]=root->element;
    ind+=1;
    InOrderTraversal(root->right_ptr,arr);
}

// binary search to find the index for query operation 
int BinarySearch(int arr[], int l, int r, int x) 
{ 
    if (r >= l) { 
        int mid = l + (r - l) / 2; 
  
        // If the element is present at the middle 
        // itself 
        if (arr[mid] == x) 
            return 1; 
  
        // If element is smaller than mid, then 
        // it can only be present in left subarray 
        else if (arr[mid] > x) 
            return BinarySearch(arr, l, mid - 1, x); 
  
        // Else the element can only be present 
        // in right subarray 
        return BinarySearch(arr, mid + 1, r, x); 
    } 
  
    // We reach here when element is not 
    // present in array 
    return -1; 
} 


//main function
// 1 . Start from here 
int main() {
	Tree_Node *root=NULL; // initialize the root node to null 
	// initializing the variables 
	char c;
	int k,a,b;
	int out;
	// running the while loops till we see char 'E' as it indicates end of input streaming
	while(c!='E'){
	    cin>>c;
	    // read the character input which could be 'I','Q','D' Or 'E'
	    if(c=='I'){
	        cin>>k; // get the element to be inserted 
	        // calling the insert function and inserting the element k into the tree is done using this function
	        root=Insert(root,k);
	    }
	    else if(c=='Q'){
	        cin>>a>>b; // get the closed interval a and b 
	        out=query(root,a,b);
	        cout<<out<<"\n";
	    }
	    else if(c=='D'){
	        cin>>k; // get the element to be deleted 
	        root=Delete(root,k);
	    }
	}
	return 0;
}
