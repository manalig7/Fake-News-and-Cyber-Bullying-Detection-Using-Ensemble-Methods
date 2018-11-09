#include<stdio.h>
#include<stdlib.h>

struct node
{
	int v1;
	int v2;
	int weight;
	struct node* next;
};
int time=0;

void insertedge(int a, int b,int c)
{
	struct node* ptr=head;
	if(head==NULL)
	{
		head=(struct node*)malloc(sizeof(struct node));
		head->v1=a;
		head->v2=b;
		head->weight=c;
		head->next=NULL;
		return;
	}
	while(ptr->next!=NULL)
	{
		ptr=ptr->next;
	}
	ptr->next=(struct node*)malloc(sizeof(struct node));
	ptr->next->v1=a;
	ptr->next->v2=b;
	ptr->next->weight=c;
	ptr->next->next=NULL;
}

int BLevel(int blevels[],int node,int weights[],int m,int arr[][3],int n)
{
	if(blevels[node]!=-1)
		return blevels[node];
	//Find the succcessors
	int max=-1;
	for(i=0;i<m;i++)
	{
		if(arr[i][0]==node)
		{
			//We found a successor arr[i][1]
			int value=BLevel(blevels,arr[i][1],weights,m,arr,n)+arr[i][2];
			if(value>max)
				max=value;
		}
	}
	max=max+weights[node];
	return max;
}

void ETF(int arr[][3],int m,int n,int rescs[][2],int weights[],int r)
{
//r is the number of resources
//For each node we calculate the b level
	int i,j;
	int blevels[n];
for(i=0;i<n;i++)
blevels[i]=-1;
//We first initialise the blevels for all the exit nodes
for(i=0;i<n;i++)
{
	flag=0;
	for(j=0;j<m;j++)
	{
		if(arr[j][0]==i)
		{
			flag=1;
			break;
		}
	}
	if(flag==0)//It is a sink node
		blevels[i]=weights[i];
}

//Now we calculate blevels for other nodes
	for(i=0;i<n;i++)
	{
		blevels[i]=BLevel(blevels,i,weights,m,arr,n);
	}

int order_of_scheduling[n];
int len=0;

//Converting the edgelist into linked list of edges
struct node* head=NULL;
for(i=0;i<m;i++)
{
	insertedge(arr[i][0],arr[i][1],arr[i][2]);
}

//////////////////////////////////////////////
//Creating a list of nodes
int nodes[n];
for(i=0;i<n;i++)
node[i]=i;

int allocation[r][2];
for(i=0;i<r;i++)
{
	allocation[i][0]=-1;
	allocation[i][1]=-1;
}
int list[n];//List of entry level nodes
int len_entry=0;
while(head!=NULL)// Till there are no edges left
{
print "Time: "+str(time)
for(i=0;i<r;i++)
{
	if(allocation[i][0]==-1)
		allocation[i][1]--;
	if(allocation[i][1]==0)
	{
		print "Task " + str(allocation[i][0])+ " completed running on resource "+str(i)+"."
		//Removing edges originating from allocation[i][0]
		while(head!=NULL&&head->v1==allocation[i][0])
		{
			struct node* temp=head;
			head=head->next;
			free(temp);
		}
		struct node* ptr=head;
		while(ptr!=NULL)
		{
			while(ptr!=NULL&&ptr->next->v1==allocation[i][0])
			{
				struct node* temp=ptr->next;
				ptr->next=ptr->next->next;
				free(ptr->next);
			}
		}
		allocation[i][0]=-1;
		allocation[i][1]=-1;

	}
}

int first=-1;
int l=-1,h=-1;
//Finding all entry level nodes
for(i=0;i<n;i++)
{
if(node[i]==-1)
	continue;
int flag=0;
struct node* ptr=head;

while(ptr!=NULL)
{
if(ptr->v2==i)
	{flag=1; break;}

ptr=ptr->next;
}

if(flag==0)//It is an entry level node
{
	if(first==-1)
	{
		first=1;
		l=len_entry;
	}
	list[len_entry]=i;
	node[i]=-1;
	len_entry++;
}

}
h=len_entry-1;
//A node becomes an entry level node only afer its predecessors have completed
//Now we have all the entry level nodes. We need to allocate to resources having earliest start time
//We first sort the nodes in the order of b level- descending 
//We only sort those nodes that have been added in this iteration
int p,k,in_idx;
for (p = l; p < h+1; p++) 
    { 
        // Find the minimum element in unsorted array 
        min_idx = p; 
        for (k = p+1; k < h+1; k++) 
          if (blevels[list[k]] > blevels[list[min_idx]]) 
            min_idx = k; 
  
        // Swap the found minimum element with the first element 
        int temp=list[min_idx];
        list[min_idx]=list[p];
        list[p]=temp;
        //swap(&arr[min_idx], &arr[i]); 
    } 

//Now allocating resource to the items in list for all empty resources
//First find how many resources are available
int num_empty=0;
for(i=0;i<r;i++)
{
	if(allocation[i][0]==-1)
		num_empty++;
}

//Now start allocating the resources and removing from the waiting list (num_empty resources can be allocated)
for(i=0;i<num_empty;i++)
{
	for(j=0;j<r;j++)
	{
		if(allocation[j][0]==-1)
		{
			allocation[j][0]=list[i];
			allocation[j][1]=weights[i];
			print "Task " + str(allocation[j][0])+ " started running on resource "+str(j)+"."
			break;
		}
	}
}
//Allocated the resources 
//Now shift the list array back by num_empty

for(i=0;i<=len_entry-num_empty-1;i++)
{
	list[i]=list[i+1];
}
len_entry=len_entry-num_empty;


time++;
}


/////////////////////////////////////////////


//Now we need to find all the entry level nodes in G and create a list of nodes in order of priority
/*
while(len<n)
{
int list[n];//List of entry level nodes
int len_entry=0;
//Finding all entry level nodes
for(i=0;i<n;i++)
{

int flag=0;
struct node* ptr=head;

while(ptr!=NULL)
{
if(ptr->v2==i)
	{flag=1; break;}

ptr=ptr->next;
}

if(flag==0)//It is an entry level node
{
	list[len_entry]=i;
	len_entry++;
}

}
//We have the list of entry level nodes
//Sort based on blevel score (decreasing order) and add to the main order

int p,k,in_idx;
for (p = 0; p < len_entry-1; p++) 
    { 
        // Find the minimum element in unsorted array 
        min_idx = p; 
        for (k = p+1; k < len_entry; k++) 
          if (blevels[list[k]] > blevels[list[min_idx]]) 
            min_idx = k; 
  
        // Swap the found minimum element with the first element 
        int temp=list[min_idx];
        list[min_idx]=list[p];
        list[p]=temp;
        //swap(&arr[min_idx], &arr[i]); 
    } 

for(p=0;p<len_entry;p++)
{
	order_of_scheduling[len]=list[p];
	len++;
}


}

return order_of_scheduling;
*/
}

void quicksort(int *node,int *burst_time,int first,int last)
{
   int i, j, pivot, temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(burst_time[i]<=burst_time[pivot]&&i<last)
            i++;
         while(burst_time[j]>burst_time[pivot])
            j--;
         if(i<j){
            temp=burst_time[i];
            burst_time[i]=burst_time[j];
            burst_time[j]=temp;
            temp=node[i];
            node[i]=node[j];
            node[j]=temp;
         }
      }

      temp=burst_time[pivot];
      burst_time[pivot]=burst_time[j];
      burst_time[j]=temp;
      temp=node[pivot];
      node[pivot]=node[j];
      node[j]=temp;
      quicksort(burst_time,first,j-1);
      quicksort(burst_time,j+1,last);

   }
}

int *SJF(int n,int *node,int *burst_time,int rescs[][2],int r)
{
	int i,j,temp,position;
	for(i = 0; i < n; i++)
      {
            position = i;
            for(j = i + 1; j < n; j++)
            {
                  if(burst_time[j] < burst_time[position])
                  {
                        position = j;
                  }
            }
            temp = burst_time[i];
            burst_time[i] = burst_time[position];
            burst_time[position] = temp;
            temp = node[i];
            node[i] = node[position];
            node[position] = temp;
      }	
int allocation[r][2];
for(i=0;i<r;i++)
{
	allocation[i][0]=-1;
	allocation[i][1]=-1;
}
int curr=0;
while(curr<n-1)
{
	print "Time: "+str(time)
	for(i=0;i<r;i++)
	{
		if(allocation[i][0]==-1)
			allocation[i][1]--;
	if(allocation[i][1]==0)
	{
		print "Task " + str(allocation[i][0])+ " completed running on resource "+str(i)+"."
		allocation[i][0]=-1;
		allocation[i][1]=-1;
	}
	}
	int num_empty=0;
	for(i=0;i<r;i++)
	{
		if(allocation[i][0]==-1)
		num_empty++;
	}

	for(i=curr;i<curr+num_empty;i++)
	{	
		for(j=0;j<r;j++)
		{
			if(allocation[j][0]==-1)
			{
				allocation[j][0]=list[i];
				allocation[j][1]=weights[i];
				print "Task " + str(allocation[j][0])+ " started running on resource "+str(j)+"."
				break;
			}
		}
	}
	curr=curr+num_empty;
	
	time++;
}

	//quicksort(node,burst_time,0,n-1);
      return node;
}


int main()
{

//Assuming nodes of DAG are named from 0 for independent tasks



	return 0;
}