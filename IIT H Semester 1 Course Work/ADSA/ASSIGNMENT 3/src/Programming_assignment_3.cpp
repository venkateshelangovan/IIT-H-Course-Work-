#include <iostream>
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace std;

/*
1. sequence_alignment
Input : 
    target : input target of type string 
     query : input query of type string 
        nt : length of the input target string of type int
        nq : length of the input query string of type int
     gcost : gap cost of type int 
     mcost : mismatch cost of type int 
    output : 2D array of type vector which stores the result of alignment score
Returns :
     output 2D array of type vector
Description : 
      This function takes the input as mentioned above and returns the 2D array of type vector which stores the
      optimal alignment score.
*/
vector< vector<int> > sequence_alignment(string target,string query,int nt,int nq,int gcost,int mcost,vector< vector<int> > output);

/*
2. BackTracking
Input : 
    target : input target of type string 
     query : input query of type string 
        nt : length of the input target string of type int
        nq : length of the input query string of type int
     gcost : gap cost of type int 
     mcost : mismatch cost of type int 
 output_dp : 2D array of type vector which stores the result of alignment score
Prints :
     optimal alignment score 
Calls : 
     print function to print the target result and query result 
Description : 
      This function takes the input as mentioned above and prints the optimal alignment score and calls the 
      function to print the target result and query result.
*/
void BackTracking(string target,string query,int nt,int nq,int gcost,int mcost,vector< vector<int> > output_dp);

/*
3. print_out
Input : 
       target_result : string of length of targetlen+querylen 
       query_result  : string of length of targetlen+querylen 
Prints : 
       target output - The pattern with gaps of the target sequence that is matched (without spaces)
       query output  - The pattern with gaps of the query sequence that is matched (without spaces)
Description : 
           This function takes the input as target_result and query_result which does certain modifications to 
           that string and returns the expected output.
*/
void print_out(string target_result,string query_result);

void print_out(string target_result,string query_result){
    // get the index from which the string to be printed as output 
    int start_index;
    for(int i=0;i<target_result.length();i++){
        // if one of the char is either not "_" break and get that index 
        if((target_result[i]!='_')||(query_result[i]!='_')){
            start_index=i;
            break;
        }
    }
    cout<<target_result.substr(start_index)<<"\n"; // substring starting from start_index
    cout<<query_result.substr(start_index)<<"\n"; // substring starting from start_index
    return;
}

vector< vector<int> > sequence_alignment(string target,string query,int nt,int nq,int gcost,int mcost,vector< vector<int> > output){
    // initializing the dynamic 2D output array 
    for(int i=0;i<=nt;i++){
        output[i][0]=i*gcost;
    }
    for(int j=0;j<=nq;j++){
        output[0][j]=j*gcost;
    }
    // to calculate the minimum penalty 
    for(int i=1;i<=nt;i++){
        for(int j=1;j<=nq;j++){
            // if target character and query character are equal 
            if(target[i-1]==query[j-1]){
                output[i][j]=output[i-1][j-1];
            }
            else{
                output[i][j]=min(output[i-1][j-1]+mcost,min(output[i-1][j]+gcost,output[i][j-1]+gcost));
            }
        }
    }
    return output; // return 2D array of type vector 
}

// function to display the output target and query string 
void BackTracking(string target,string query,int nt,int nq,int gcost,int mcost,vector< vector<int> > output_dp){
    // initialize the variables 
    int output_len=nt+nq; // output len to store the output target and query
    int t=nt,q=nq; // to iterate through the target and query
    string target_out(output_len+1,'_'); // initialize the output target string 
    string query_out(output_len+1,'_'); // initialize the output query string 
    int i=output_len,j=output_len;
    // Steps involved in backtracking and store the results in target_out and query_out 
    while(!((t==0)||(q==0))){
        if(target[t-1]==query[q-1]){
            // if target and query character are equal, store the char in output string 
            target_out[i]=target[t-1];
            query_out[j]=query[q-1];
            i-=1;
            j-=1;
            t-=1;
            q-=1;
        }
        else if(output_dp[t][q-1]+gcost==output_dp[t][q]){
            // we first perform this to confirm lexicographically the least for query
            query_out[j]=query[q-1];
            i-=1;
            j-=1;
            q-=1;
        }
        else if(output_dp[t-1][q]+gcost==output_dp[t][q]){
            target_out[i]=target[t-1];
            i-=1;
            j-=1;
            t-=1;
        }

        else if(output_dp[t-1][q-1]+mcost==output_dp[t][q]){
            target_out[i]=target[t-1];
            query_out[j]=query[q-1];
            i-=1;
            j-=1;
            t-=1;
            q-=1;
        }
    }
    // copy the remaining content of target string to output target 
    while(i>0){
        target_out[i--]=(t>0)?target[--t] : '_';
    }
    // copy the remaining content of query string to output query 
    while(j>0){
        query_out[j--]=(q>0)?query[--q]:'_';
    }
    // calls the function print_out to get target_output and query_out 
    print_out(target_out,query_out);
    return ;
}
int main() {
	int target_length,query_length,gap_cost,mismatch_cost;
	string target_input,query_input;
	cin>>target_length>>query_length; // get the input of target length and query length 
	cin>>gap_cost>>mismatch_cost; // get the input of gap cost and mismatch cost
	cin>>target_input; // get the target input of type string 
	cin>>query_input; // get the query input of type string 
	std::string target = target_input.substr(0,target_length);
	std::string query = query_input.substr(0,query_length);
    vector< vector<int> >sequence_dp(target_length+1, vector<int>(query_length+1)); // 2D Array to store alignment score
	sequence_dp=sequence_alignment(target,query,target_length,query_length,gap_cost,mismatch_cost,sequence_dp);
	//prints the optimal alignment score 
	cout<<sequence_dp[target_length][query_length]<<"\n";
	// BackTracking to get the target and query output
	BackTracking(target,query,target_length,query_length,gap_cost,mismatch_cost,sequence_dp);
	return 0;
}
